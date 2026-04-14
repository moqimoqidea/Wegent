// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

import '@testing-library/jest-dom'
import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { ErrorCard } from '@/features/tasks/components/message/ErrorCard'
import type { Message } from '@/features/tasks/components/message/MessageBubble'

jest.mock('@/hooks/useTranslation', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string>) => options?.model ?? key,
  }),
}))

jest.mock('@/features/common/UserContext', () => ({
  useUser: () => ({
    user: {
      id: 1,
      user_name: 'tester',
    },
  }),
}))

jest.mock('@/features/tasks/hooks/useErrorRecommendations', () => ({
  useErrorRecommendations: () => ({
    getRecommendedModels: () => [
      {
        name: 'gemini-2.5-pro',
        displayName: 'Gemini 2.5 Pro',
        provider: 'google',
        modelId: 'gemini-2.5-pro',
        type: 'shared',
      },
    ],
  }),
}))

function createErrorMessage(timestamp: number, error: string): Message {
  return {
    type: 'ai',
    content: '',
    timestamp,
    subtaskId: 42,
    status: 'error',
    error,
    errorType: 'model_unavailable',
  }
}

describe('ErrorCard', () => {
  beforeEach(() => {
    localStorage.clear()
    jest.clearAllMocks()
  })

  it('reopens a new last-error instance after switch-model retry fails again', async () => {
    const user = userEvent.setup()
    const onRetryWithModel = jest.fn()

    const { rerender } = render(
      <ErrorCard
        error="original model failed"
        errorType="model_unavailable"
        subtaskId={42}
        taskId={100}
        timestamp={1000}
        message={createErrorMessage(1000, 'original model failed')}
        isLastErrorMessage={true}
        onRetryWithModel={onRetryWithModel}
      />
    )

    await user.click(screen.getByTestId('error-card-model-recommend-gemini-2.5-pro'))

    expect(onRetryWithModel).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId('error-card-collapsed')).toBeInTheDocument()

    rerender(
      <ErrorCard
        error="fallback model also failed"
        errorType="model_unavailable"
        subtaskId={42}
        taskId={100}
        timestamp={2000}
        message={createErrorMessage(2000, 'fallback model also failed')}
        isLastErrorMessage={true}
        onRetryWithModel={onRetryWithModel}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId('error-card')).toBeInTheDocument()
    })

    expect(screen.queryByTestId('error-card-collapsed')).not.toBeInTheDocument()
  })
})