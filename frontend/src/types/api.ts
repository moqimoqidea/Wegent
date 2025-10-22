// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Authentication Types


// User Types
export interface User {
  id: number
  user_name: string
  email: string
  is_active: boolean
  created_at: string
  updated_at: string
  git_info: GitInfo[]
}


/** Git account information */
export interface GitInfo {
  git_domain: string
  git_token: string
  /** Type: "github" | "gitlab" */
  type: 'github' | 'gitlab'
}

// Bot Types
export interface Bot {
  id: number
  name: string
  agent_name: string
  agent_config: Record<string, any>
  system_prompt: string
  mcp_servers: Record<string, any>
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface Team {
  id: number
  name: string
  description: string
  bots: TeamBot[]
  workflow: Record<string, any>
  is_active: boolean
  user_id: number
  created_at: string
  updated_at: string
  share_status?: number // 0: 个人团队, 1: 分享中, 2: 共享团队
  user?: {
    user_name: string
  }
}


/** Bot information (used for Team.bots) */
export interface TeamBot {
  bot_id: number
  bot_prompt: string
  role?: string
}

/** TaskDetail structure (adapted to latest backend response) */
// Task Types
export type TaskStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED' | 'DELETE'
export interface TaskDetail {
  id: number
  title: string
  git_url: string
  git_repo: string
  git_repo_id: number
  git_domain: string
  branch_name: string
  prompt: string
  status: TaskStatus
  progress: number
  batch: number
  result: Record<string, any>
  error_message: string
  created_at: string
  // updated_at: string
  // completed_at: string
  user: User
  team: Team
  subtasks: TaskDetailSubtask[]
}

/** Subtask structure (adapted to latest backend response) */
export interface TaskDetailSubtask {
  task_id: number
  team_id: number
  title: string
  /** Multi-bot support */
  bot_ids: number[]
  /** Role */
  role: string
  /** Message ID */
  message_id: number
  /** Parent Task ID */
  parent_id: number
  prompt: string
  executor_namespace: string
  executor_name: string
  status: TaskStatus
  progress: number
  batch: number
  result: Record<string, any> | { thinking?: any[], value?: string }
  error_message: string
  id: number
  user_id: number
  created_at: string
  updated_at: string
  completed_at: string
  bots: Bot[]
}

export interface Task {
  id: number
  title: string
  team_id: number
  git_url: string
  git_repo: string
  git_repo_id: number
  git_domain: string
  branch_name: string
  prompt: string
  status: TaskStatus
  progress: number
  batch: number
  result: Record<string, any>
  error_message: string
  user_id: number
  user_name: string
  created_at: string
  updated_at: string
  completed_at: string
}




/** GitHub repository new structure */
export interface GitRepoInfo {
  git_repo_id: number
  name: string
  git_repo: string
  git_url: string
  git_domain: string
  private: boolean
  /** Type: "github" | "gitlab" */
  type: 'github' | 'gitlab'
}


export interface GitBranch {
  name: string
  protected: boolean
  default: boolean
}

// Common API Response Types
export interface APIError {
  message: string
  detail?: string
}

export interface SuccessMessage {
  message: string
}

// Pagination Types
export interface PaginationParams {
  page?: number
  limit?: number
}
