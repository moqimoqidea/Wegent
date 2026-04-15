// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect, useCallback, useRef } from 'react'

import { modelApis, type UnifiedModel } from '@/apis/models'
import type { ErrorRecommendationEntry } from '@/apis/models'

const CACHE_TTL_MS = 5 * 60 * 1000 // 5 minutes

interface CacheEntry {
  data: Record<string, ErrorRecommendationEntry>
  timestamp: number
}

export const DEFAULT_ERROR_RECOMMENDATION_KEY = 'default_errors'

const ERROR_TYPE_ALIASES: Record<string, string[]> = {
  model_unavailable: ['llm_error'],
  llm_error: ['model_unavailable'],
}

// Module-level cache shared across all hook instances
let cachedEntry: CacheEntry | null = null

function hasRecommendationEntry(
  recommendations: Record<string, ErrorRecommendationEntry>,
  key: string
): boolean {
  return Object.prototype.hasOwnProperty.call(recommendations, key)
}

function getRecommendationLookupKeys(errorType: string): string[] {
  const normalizedErrorType = errorType.trim()
  if (!normalizedErrorType) {
    return []
  }

  return Array.from(
    new Set([normalizedErrorType, ...(ERROR_TYPE_ALIASES[normalizedErrorType] ?? [])])
  )
}

export function getRecommendedModelsForError(
  recommendations: Record<string, ErrorRecommendationEntry>,
  errorType: string
): UnifiedModel[] {
  for (const key of getRecommendationLookupKeys(errorType)) {
    if (hasRecommendationEntry(recommendations, key)) {
      return recommendations[key]?.models ?? []
    }
  }

  return recommendations[DEFAULT_ERROR_RECOMMENDATION_KEY]?.models ?? []
}

/**
 * Hook to fetch error-type-specific model recommendations.
 *
 * Caches results for 5 minutes. Falls back to empty recommendations
 * if the API is unavailable.
 */
export function useErrorRecommendations() {
  const [recommendations, setRecommendations] = useState<Record<string, ErrorRecommendationEntry>>(
    cachedEntry?.data ?? {}
  )
  const [isLoading, setIsLoading] = useState(false)
  const fetchedRef = useRef(false)

  useEffect(() => {
    if (fetchedRef.current) return

    // Use cache if still valid
    if (cachedEntry && Date.now() - cachedEntry.timestamp < CACHE_TTL_MS) {
      setRecommendations(cachedEntry.data)
      return
    }

    fetchedRef.current = true
    setIsLoading(true)

    modelApis
      .getErrorRecommendations()
      .then(response => {
        const data = response.data || {}
        cachedEntry = { data, timestamp: Date.now() }
        setRecommendations(data)
      })
      .catch(() => {
        // API unavailable — fallback to empty recommendations
        setRecommendations({})
      })
      .finally(() => {
        setIsLoading(false)
      })
  }, [])

  const getRecommendedModels = useCallback(
    (errorType: string): UnifiedModel[] => getRecommendedModelsForError(recommendations, errorType),
    [recommendations]
  )

  return { recommendations, getRecommendedModels, isLoading }
}
