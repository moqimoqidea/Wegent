// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * User module: encapsulates login/logout/getCurrentUser/updateUser and token management.
 * All types and logic are self-contained for cohesion.
 */

import type { GitInfo, User } from '@/types/api'

// Type definitions
export interface LoginRequest {
  user_name: string
  password: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
}

export interface UpdateUserRequest {
  user_name?: string
  email?: string
  is_active?: boolean
  git_info?: GitInfo[]
}


const TOKEN_KEY = 'auth_token'
const TOKEN_EXPIRE_KEY = 'auth_token_expire'

function getJwtExp(token: string): number | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    return typeof payload.exp === 'number' ? payload.exp * 1000 : null // convert to milliseconds
  } catch {
    return null
  }
}

export function setToken(token: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem(TOKEN_KEY, token)
    const exp = getJwtExp(token)
    if (exp) {
      localStorage.setItem(TOKEN_EXPIRE_KEY, String(exp))
    } else {
      localStorage.removeItem(TOKEN_EXPIRE_KEY)
    }
  }
}

export function getToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem(TOKEN_KEY)
  }
  return null
}

export function getTokenExpire(): number | null {
  if (typeof window !== 'undefined') {
    const exp = localStorage.getItem(TOKEN_EXPIRE_KEY)
    return exp ? Number(exp) : null
  }
  return null
}

export function removeToken() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(TOKEN_EXPIRE_KEY)
  }
}

/**
 * Check if token exists and is not expired
 */
function isAuthenticated(): boolean {
  const token = getToken()
  const exp = getTokenExpire()
  if (!token || !exp) return false
  return Date.now() < exp
}

// API Client
import { apiClient } from './client'
import { paths } from '@/config/paths'

export const userApis = {
  async login(data: LoginRequest): Promise<User> {
    const res: LoginResponse = await apiClient.post("/auth/login", data)
    setToken(res.access_token)
    // Get user information after login
    return await apiClient.get('/users/me')
  },

  logout() {
    removeToken()
    if (typeof window !== 'undefined') {
      window.location.href = paths.home.getHref()
    }
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get('/users/me')
  },

  async updateUser(data: UpdateUserRequest): Promise<User> {
    return apiClient.put('/users/me', data)
  },

  isAuthenticated(): boolean {
    return isAuthenticated()
  }
}

export async function loginWithOidcToken(accessToken: string): Promise<void> {
  setToken(accessToken)
}