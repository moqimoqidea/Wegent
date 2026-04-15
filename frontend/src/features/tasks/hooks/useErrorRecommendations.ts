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

const ERROR_TYPE_ALIASES: Record<string, string[]> = {
  model_unavailable: ['llm_error'],
  llm_error: ['model_unavailable'],
}

// Module-level cache shared across all hook instances
let cachedEntry: CacheEntry | null = null

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
    (errorType: string): UnifiedModel[] => {
      const candidateKeys = [errorType, ...(ERROR_TYPE_ALIASES[errorType] ?? [])]

      for (const key of candidateKeys) {
        const models = recommendations[key]?.models
        if (models?.length) {
          return models
        }
      }

      return []
    },
    [recommendations]
  )

  return { recommendations, getRecommendedModels, isLoading }
}
