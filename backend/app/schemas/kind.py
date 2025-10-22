# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kubernetes-style API schemas for cloud-native agent management
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ObjectMeta(BaseModel):
    """Standard Kubernetes object metadata"""
    name: str
    namespace: str = "default"
    labels: Optional[Dict[str, str]] = None
    # annotations: Optional[Dict[str, str]] = None


class Status(BaseModel):
    """Standard status object"""
    state: str
    message: Optional[str] = None
    # conditions: Optional[List[Dict[str, Any]]] = None


# Ghost CRD schemas
class GhostSpec(BaseModel):
    """Ghost specification"""
    systemPrompt: str
    mcpServers: Optional[Dict[str, Any]] = None


class GhostStatus(Status):
    """Ghost status"""
    state: str = "Available"  # Available, Unavailable


class Ghost(BaseModel):
    """Ghost CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Ghost"
    metadata: ObjectMeta
    spec: GhostSpec
    status: Optional[GhostStatus] = None


class GhostList(BaseModel):
    """Ghost list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "GhostList"
    items: List[Ghost]


# Model CRD schemas
class ModelSpec(BaseModel):
    """Model specification"""
    modelConfig: Dict[str, Any]


class ModelStatus(Status):
    """Model status"""
    state: str = "Available"  # Available, Unavailable


class Model(BaseModel):
    """Model CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Model"
    metadata: ObjectMeta
    spec: ModelSpec
    status: Optional[ModelStatus] = None


class ModelList(BaseModel):
    """Model list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "ModelList"
    items: List[Model]


# Shell CRD schemas
class ModelRef(BaseModel):
    """Reference to a Model"""
    name: str
    namespace: str = "default"


class ShellSpec(BaseModel):
    """Shell specification"""
    runtime: str
    supportModel: Optional[List[str]] = None


class ShellStatus(Status):
    """Shell status"""
    state: str = "Available"  # Available, Unavailable


class Shell(BaseModel):
    """Shell CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Shell"
    metadata: ObjectMeta
    spec: ShellSpec
    status: Optional[ShellStatus] = None


class ShellList(BaseModel):
    """Shell list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "ShellList"
    items: List[Shell]


# Bot CRD schemas
class GhostRef(BaseModel):
    """Reference to a Ghost"""
    name: str
    namespace: str = "default"


class ShellRef(BaseModel):
    """Reference to a Shell"""
    name: str
    namespace: str = "default"


class BotSpec(BaseModel):
    """Bot specification"""
    ghostRef: GhostRef
    shellRef: ShellRef
    modelRef: ModelRef


class BotStatus(Status):
    """Bot status"""
    state: str = "Available"  # Available, Unavailable


class Bot(BaseModel):
    """Bot CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Bot"
    metadata: ObjectMeta
    spec: BotSpec
    status: Optional[BotStatus] = None


class BotList(BaseModel):
    """Bot list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "BotList"
    items: List[Bot]


# Team CRD schemas
class BotTeamRef(BaseModel):
    """Reference to a Bot in Team"""
    name: str
    namespace: str = "default"


class TeamMember(BaseModel):
    """Team member specification"""
    botRef: BotTeamRef
    prompt: Optional[str] = None
    role: Optional[str] = None


class TeamSpec(BaseModel):
    """Team specification"""
    members: List[TeamMember]
    collaborationModel: str  # pipeline、route、coordinate、collaborate


class TeamStatus(Status):
    """Team status"""
    state: str = "Available"  # Available, Unavailable


class Team(BaseModel):
    """Team CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Team"
    metadata: ObjectMeta
    spec: TeamSpec
    status: Optional[TeamStatus] = None


class TeamList(BaseModel):
    """Team list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "TeamList"
    items: List[Team]


# Workspace CRD schemas
class Repository(BaseModel):
    """Repository configuration"""
    gitUrl: str
    gitRepo: str
    gitRepoId: Optional[int] = None
    branchName: str
    gitDomain: str

class WorkspaceSpec(BaseModel):
    """Workspace specification"""
    repository: Repository


class WorkspaceStatus(Status):
    """Workspace status"""
    state: str = "Available"  # Available, Unavailable


class Workspace(BaseModel):
    """Workspace CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Workspace"
    metadata: ObjectMeta
    spec: WorkspaceSpec
    status: Optional[WorkspaceStatus] = None


class WorkspaceList(BaseModel):
    """Workspace list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "WorkspaceList"
    items: List[Workspace]


# Task CRD schemas
class TeamTaskRef(BaseModel):
    """Reference to a Team"""
    name: str
    namespace: str = "default"


class WorkspaceTaskRef(BaseModel):
    """Reference to a Workspace"""
    name: str
    namespace: str = "default"


class TaskSpec(BaseModel):
    """Task specification"""
    title: str
    prompt: str
    teamRef: TeamTaskRef
    workspaceRef: WorkspaceTaskRef


class TaskStatus(Status):
    """Task status"""
    state: str = "Available"  # Available, Unavailable
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, DELETE
    progress: int = 0
    result: Optional[Dict[str, Any]] = None
    errorMessage: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    subTasks:  Optional[List[Dict[str, Any]]] = None 


class Task(BaseModel):
    """Task CRD"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "Task"
    metadata: ObjectMeta
    spec: TaskSpec
    status: Optional[TaskStatus] = None


class TaskList(BaseModel):
    """Task list"""
    apiVersion: str = "agent.wecode.io/v1"
    kind: str = "TaskList"
    items: List[Task]


class BatchResponse(BaseModel):
    """Batch operation response"""
    success: bool
    message: str
    results: List[Dict[str, Any]]