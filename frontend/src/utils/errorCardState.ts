// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Error card interaction state persistence via localStorage.
 * Tracks which error cards the user has interacted with (clicked a solution).
 */

const STORAGE_KEY = 'wegent_error_card_interactions'
const MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000 // 30 days

interface ErrorCardInteraction {
  interactedAt: number
  action: string
}

type InteractionStore = Record<string, ErrorCardInteraction>

function readStore(): InteractionStore {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return {}
    return JSON.parse(raw) as InteractionStore
  } catch {
    return {}
  }
}

function writeStore(store: InteractionStore): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store))
  } catch {
    // localStorage full or unavailable — silently ignore
  }
}

/**
 * Mark an error card as interacted (user clicked a solution).
 */
export function markErrorInteracted(subtaskId: number, action: string): void {
  const store = readStore()
  store[String(subtaskId)] = { interactedAt: Date.now(), action }
  writeStore(store)
}

/**
 * Check if an error card has been interacted with.
 */
export function isErrorInteracted(subtaskId: number): boolean {
  const store = readStore()
  return String(subtaskId) in store
}

/**
 * Remove stale entries older than MAX_AGE_MS.
 * Called lazily — no need for a timer.
 */
export function cleanupStaleEntries(): void {
  const store = readStore()
  const cutoff = Date.now() - MAX_AGE_MS
  let changed = false

  for (const key of Object.keys(store)) {
    if (store[key].interactedAt < cutoff) {
      delete store[key]
      changed = true
    }
  }

  if (changed) writeStore(store)
}
