#!/usr/bin/env python

# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

import asyncio
import threading
import time
from typing import Dict, Optional, Any, Tuple, List, ClassVar
from dataclasses import dataclass

from shared.logger import setup_logger
from shared.status import TaskStatus
from executor.agents import Agent, AgentFactory
from executor.agents.claude_code.claude_code_agent import ClaudeCodeAgent
from executor.agents.agno.agno_agent import AgnoAgent

logger = setup_logger("agent_service")

def _format_task_log(task_id, subtask_id):
    return f"task_id: {task_id}.{subtask_id}"


@dataclass
class AgentSession:
    agent: Agent
    created_at: float


class AgentService:
    _instance: ClassVar[Optional['AgentService']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._agent_sessions = {}
        return cls._instance

    def get_agent(self, agent_session_id: str) -> Optional[Agent]:
        session = self._agent_sessions.get(agent_session_id)
        return session.agent if session else None

    def _generate_agent_session_id(self, task_id: Any, subtask_id: Any) -> str:
        """Generate a unique session ID for an agent based on task and subtask IDs."""
        return f"agent_session_{task_id}"

    def create_agent(self, task_data: Dict[str, Any]) -> Optional[Agent]:
        task_id = task_data.get("task_id", -1)
        subtask_id = task_data.get("subtask_id", -1)

        if (existing_agent := self.get_agent(self._generate_agent_session_id(task_id, subtask_id))):
            logger.info(f"[{_format_task_log(task_id, subtask_id)}] Reusing existing agent")
            return existing_agent

        try:
            bot_config = task_data.get("bot")
            if isinstance(bot_config, list):
                agent_name = bot_config[0].get("agent_name", "").strip().lower()
            else:
                agent_name = bot_config.get("agent_name", "").strip().lower()

            logger.info(f"[{_format_task_log(task_id, subtask_id)}] Creating new agent '{agent_name}'")
            agent = AgentFactory.get_agent(agent_name, task_data)

            if not agent:
                logger.error(f"[{_format_task_log(task_id, subtask_id)}] Failed to create agent")
                return None
                
            init_status = agent.initialize()
            if init_status != TaskStatus.SUCCESS:
                logger.error(f"[{_format_task_log(task_id, subtask_id)}] Failed to initialize agent: {init_status}")
                return None

            self._agent_sessions[task_id] = AgentSession(agent=agent, created_at=time.time())
            return agent

        except Exception as e:
            logger.exception(f"[{_format_task_log(task_id, subtask_id)}] Exception during agent creation: {e}")
            return None

    def execute_agent_task(self, agent: Agent, pre_executed: Optional[TaskStatus] = None) -> Tuple[TaskStatus, Optional[str]]:
        try:
            logger.info(f"[{agent.get_name()}][{_format_task_log(agent.task_id, agent.subtask_id)}] Executing with pre_executed={pre_executed}")
            return agent.handle(pre_executed)
        except Exception as e:
            logger.exception(f"[{agent.get_name()}][{_format_task_log(agent.task_id, agent.subtask_id)}] Execution error: {e}")
            return TaskStatus.FAILED, str(e)

    def execute_task(self, task_data: Dict[str, Any]) -> Tuple[TaskStatus, Optional[str]]:
        task_id = task_data.get("task_id", -1)
        subtask_id = task_data.get("subtask_id", -1)
        try:
            agent = self.get_agent(self._generate_agent_session_id(task_id, subtask_id))
            
            # If agent exists, update prompt
            if agent and hasattr(agent, 'update_prompt') and "prompt" in task_data:
                new_prompt = task_data.get("prompt", "")
                logger.info(f"[{_format_task_log(task_id, subtask_id)}] Updating prompt for existing agent")
                agent.update_prompt(new_prompt)
            # If agent doesn't exist, create new agent
            elif not agent:
                agent = self.create_agent(task_data)
                
            if not agent:
                msg = f"[{_format_task_log(task_id, subtask_id)}] Unable to get or create agent"
                logger.error(msg)
                return TaskStatus.FAILED, msg
            return self.execute_agent_task(agent)
        except Exception as e:
            logger.exception(f"[{_format_task_log(task_id, subtask_id)}] Task execution error: {e}")
            return TaskStatus.FAILED, str(e)

    async def _close_agent_session(self, task_id: str, agent: Agent) -> Tuple[TaskStatus, Optional[str]]:
        try:
            agent_name = agent.get_name()
            if agent_name == "ClaudeCodeAgent":
                await ClaudeCodeAgent.close_client(task_id)
                logger.info(f"[{_format_task_log(task_id, -1)}] Closed Claude client")
            elif agent_name == "Agno":
                await AgnoAgent.close_client(task_id)
                logger.info(f"[{_format_task_log(task_id, -1)}] Closed Agno client")
            return TaskStatus.SUCCESS, None
        except Exception as e:
            logger.exception(f"[{_format_task_log(task_id, -1)}] Error closing agent: {e}")
            return TaskStatus.FAILED, str(e)

    async def delete_session_async(self, task_id: str) -> Tuple[TaskStatus, Optional[str]]:
        session = self._agent_sessions.get(task_id)
        if not session:
            return TaskStatus.FAILED, f"[{_format_task_log(task_id, -1)}] No session found"

        try:
            status, error_msg = await self._close_agent_session(task_id, session.agent)
            if status != TaskStatus.SUCCESS:
                return status, error_msg
            del self._agent_sessions[task_id]
            return TaskStatus.SUCCESS, f"[{_format_task_log(task_id, -1)}] Session deleted"
        except Exception as e:
            logger.exception(f"[{task_id}] Error deleting session: {e}")
            return TaskStatus.FAILED, str(e)

    def delete_session(self, task_id: str) -> Tuple[TaskStatus, Optional[str]]:
        try:
            return asyncio.run(self.delete_session_async(task_id))
        except RuntimeError as e:
            if "already running" in str(e):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.delete_session_async(task_id))
            logger.exception(f"[{task_id}] Runtime error deleting session: {e}")
            return TaskStatus.FAILED, str(e)
        except Exception as e:
            logger.exception(f"[{task_id}] Unexpected error deleting session: {e}")
            return TaskStatus.FAILED, str(e)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "task_id": task_id,
                "agent_type": session.agent.get_name(),
                "pre_executed": session.agent.pre_executed,
                "created_at": session.created_at
            }
            for task_id, session in self._agent_sessions.items()
        ]

    async def _close_claude_sessions(self) -> Tuple[TaskStatus, Optional[str]]:
        try:
            await ClaudeCodeAgent.close_all_clients()
            logger.info("Closed all Claude client connections")
            return TaskStatus.SUCCESS, None
        except Exception as e:
            logger.exception("Error closing Claude client connections")
            return TaskStatus.FAILED, str(e)

    async def _close_agno_sessions(self) -> Tuple[TaskStatus, Optional[str]]:
        try:
            await AgnoAgent.close_all_clients()
            logger.info("Closed all Agno client connections")
            return TaskStatus.SUCCESS, None
        except Exception as e:
            logger.exception("Error closing Agno client connections")
            return TaskStatus.FAILED, str(e)

    async def close_all_agent_sessions(self) -> Tuple[TaskStatus, str, Dict[str, str]]:
        results: List[str] = []
        errors: List[str] = []
        error_detail: Dict[str, str] = {}
        agent_types = {s.agent.get_name() for s in self._agent_sessions.values()}

        if "ClaudeCodeAgent" in agent_types:
            status, msg = await self._close_claude_sessions()
            if status == TaskStatus.SUCCESS:
                results.append("Claude")
            else:
                errors.append("Claude")
                error_detail["ClaudeCodeAgent"] = msg or "Unknown error"

        if "Agno" in agent_types:
            status, msg = await self._close_agno_sessions()
            if status == TaskStatus.SUCCESS:
                results.append("Agno")
            else:
                errors.append("Agno")
                error_detail["AgnoAgent"] = msg or "Unknown error"

        self._agent_sessions.clear()

        if not errors:
            return TaskStatus.SUCCESS, "All agent sessions closed successfully", {}
        else:
            message = f"Some agents failed to close: {', '.join(errors)}; Successful: {', '.join(results) or 'None'}"
            return TaskStatus.FAILED, message, error_detail