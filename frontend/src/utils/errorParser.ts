// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Parse error messages and return user-friendly error information.
 *
 * Supports backend-provided error types (preferred) with string-based
 * keyword matching as fallback.
 */

export type ErrorType =
  | 'context_length_exceeded'
  | 'quota_exceeded'
  | 'rate_limit'
  | 'payload_too_large'
  | 'network_error'
  | 'timeout_error'
  | 'llm_error'
  | 'llm_unsupported'
  | 'invalid_parameter'
  | 'forbidden'
  | 'container_oom'
  | 'container_error'
  | 'generic_error'

export interface ParsedError {
  type: ErrorType
  message: string
  originalError?: string
  retryable?: boolean
}

// Valid backend error type values
const VALID_BACKEND_TYPES = new Set<string>([
  'context_length_exceeded',
  'quota_exceeded',
  'rate_limit',
  'model_unavailable',
  'container_oom',
  'container_error',
  'network_error',
  'timeout_error',
  'llm_unsupported',
  'forbidden',
  'payload_too_large',
  'invalid_parameter',
  'generic_error',
])

// Map backend error codes to frontend ErrorType
const BACKEND_TYPE_MAP: Record<string, ErrorType> = {
  context_length_exceeded: 'context_length_exceeded',
  quota_exceeded: 'quota_exceeded',
  rate_limit: 'rate_limit',
  model_unavailable: 'llm_error',
  container_oom: 'container_oom',
  container_error: 'container_error',
  network_error: 'network_error',
  timeout_error: 'timeout_error',
  llm_unsupported: 'llm_unsupported',
  forbidden: 'forbidden',
  payload_too_large: 'payload_too_large',
  invalid_parameter: 'invalid_parameter',
  generic_error: 'generic_error',
}

// Retryability by error type
const RETRYABLE_MAP: Record<ErrorType, boolean> = {
  context_length_exceeded: false,
  quota_exceeded: false,
  rate_limit: true,
  payload_too_large: true,
  network_error: true,
  timeout_error: true,
  llm_error: true,
  llm_unsupported: false,
  invalid_parameter: true,
  forbidden: false,
  container_oom: false,
  container_error: false,
  generic_error: true,
}

/**
 * Parse error and return structured error information.
 *
 * @param error - Error object or error message
 * @param backendType - Optional error type from backend classification
 * @returns Parsed error information
 */
export function parseError(error: Error | string, backendType?: string): ParsedError {
  const errorMessage = typeof error === 'string' ? error : error.message

  // Use backend-provided type if valid
  if (backendType && VALID_BACKEND_TYPES.has(backendType)) {
    const type = BACKEND_TYPE_MAP[backendType] || 'generic_error'
    return {
      type,
      message: errorMessage,
      originalError: errorMessage,
      retryable: RETRYABLE_MAP[type],
    }
  }

  // Fall back to keyword-based string matching
  return classifyByMessage(errorMessage)
}

/**
 * Classify error by keyword matching on the message string.
 */
function classifyByMessage(errorMessage: string): ParsedError {
  const lowerMessage = errorMessage.toLowerCase()

  // Context length exceeded (check before general LLM errors)
  if (
    lowerMessage.includes('prompt is too long') ||
    lowerMessage.includes('context_length_exceeded') ||
    lowerMessage.includes('context length exceeded') ||
    lowerMessage.includes('maximum context length') ||
    lowerMessage.includes('token limit exceeded') ||
    lowerMessage.includes('tokens exceeds the model') ||
    lowerMessage.includes('input is too long') ||
    lowerMessage.includes('maximum number of tokens')
  ) {
    return buildResult('context_length_exceeded', errorMessage)
  }

  // Container OOM
  if (
    lowerMessage.includes('out of memory') ||
    lowerMessage.includes('oom') ||
    lowerMessage.includes('memory allocation')
  ) {
    return buildResult('container_oom', errorMessage)
  }

  // Container/executor errors
  if (
    lowerMessage.includes('container') ||
    lowerMessage.includes('executor') ||
    lowerMessage.includes('docker') ||
    lowerMessage.includes('disappeared unexpectedly') ||
    lowerMessage.includes('no ports mapped') ||
    lowerMessage.includes('crashed unexpectedly') ||
    lowerMessage.includes('exit code')
  ) {
    return buildResult('container_error', errorMessage)
  }

  // Quota exceeded (check before rate_limit — more specific)
  if (
    lowerMessage.includes('quota exceeded') ||
    lowerMessage.includes('insufficient_quota') ||
    lowerMessage.includes('billing') ||
    lowerMessage.includes('credit balance') ||
    lowerMessage.includes('payment required') ||
    lowerMessage.includes('insufficient funds') ||
    lowerMessage.includes('exceeded your current quota')
  ) {
    return buildResult('quota_exceeded', errorMessage)
  }

  // Rate limit (temporary throttling)
  if (
    lowerMessage.includes('rate limit') ||
    lowerMessage.includes('rate_limit') ||
    lowerMessage.includes('too many requests') ||
    lowerMessage.includes('throttl')
  ) {
    return buildResult('rate_limit', errorMessage)
  }

  // Forbidden/unauthorized
  if (
    lowerMessage.includes('forbidden') ||
    lowerMessage.includes('not allowed') ||
    lowerMessage.includes('unauthorized') ||
    lowerMessage.includes('403')
  ) {
    return buildResult('forbidden', errorMessage)
  }

  // Model unsupported (multi-modal, incompatibility)
  if (
    lowerMessage.includes('multi-modal') ||
    lowerMessage.includes('multimodal') ||
    lowerMessage.includes('do not support') ||
    lowerMessage.includes('does not support') ||
    lowerMessage.includes('not support image') ||
    (lowerMessage.includes('llm model') && lowerMessage.includes('received'))
  ) {
    return buildResult('llm_unsupported', errorMessage)
  }

  // General LLM errors (model unavailable, API errors)
  if (
    lowerMessage.includes('model not found') ||
    lowerMessage.includes('model unavailable') ||
    lowerMessage.includes('llm request failed') ||
    lowerMessage.includes('llm api error') ||
    lowerMessage.includes('llm call failed') ||
    lowerMessage.includes('llm service error') ||
    lowerMessage.includes('model error') ||
    lowerMessage.includes('api rate limit') ||
    lowerMessage.includes('token limit')
  ) {
    return buildResult('llm_error', errorMessage)
  }

  // Invalid parameter
  if (lowerMessage.includes('invalid') && lowerMessage.includes('parameter')) {
    return buildResult('invalid_parameter', errorMessage)
  }

  // Payload too large
  if (lowerMessage.includes('413') || lowerMessage.includes('payload too large')) {
    return buildResult('payload_too_large', errorMessage)
  }

  // Network errors
  if (
    lowerMessage.includes('network') ||
    lowerMessage.includes('fetch') ||
    lowerMessage.includes('connection') ||
    lowerMessage.includes('not connected') ||
    lowerMessage.includes('websocket')
  ) {
    return buildResult('network_error', errorMessage)
  }

  // Timeout
  if (lowerMessage.includes('timeout') || lowerMessage.includes('timed out')) {
    return buildResult('timeout_error', errorMessage)
  }

  // Generic fallback
  return buildResult('generic_error', errorMessage)
}

function buildResult(type: ErrorType, errorMessage: string): ParsedError {
  return {
    type,
    message: errorMessage,
    originalError: errorMessage,
    retryable: RETRYABLE_MAP[type],
  }
}

/**
 * Get user-friendly error message with i18n support.
 *
 * @param error - Error object or error message
 * @param t - i18n translation function
 * @param backendType - Optional error type from backend classification
 * @returns User-friendly error message
 */
export function getUserFriendlyErrorMessage(
  error: Error | string,
  t: (key: string) => string,
  backendType?: string
): string {
  const parsed = parseError(error, backendType)

  switch (parsed.type) {
    case 'context_length_exceeded':
      return t('errors.context_length_exceeded')
    case 'quota_exceeded':
      return t('errors.quota_exceeded')
    case 'rate_limit':
      return t('errors.rate_limit')
    case 'forbidden':
      return t('errors.forbidden') || t('errors.generic_error')
    case 'container_oom':
      return t('errors.container_oom')
    case 'container_error':
      return t('errors.container_error')
    case 'llm_unsupported':
      return t('errors.llm_unsupported')
    case 'llm_error':
      return t('errors.llm_error')
    case 'invalid_parameter':
      return t('errors.invalid_parameter')
    case 'payload_too_large':
      return t('errors.payload_too_large')
    case 'network_error':
      return t('errors.network_error')
    case 'timeout_error':
      return t('errors.timeout_error')
    default:
      return t('errors.generic_error')
  }
}

/**
 * Get error message for display in toast/UI.
 *
 * Logic:
 * - For specific error types, return friendly translated message
 * - For generic/unclassified errors, return the original error message directly
 *
 * @param error - Error object or error message
 * @param t - i18n translation function
 * @param fallbackMessage - Fallback message when originalError is empty
 * @param backendType - Optional error type from backend classification
 * @returns Display message for toast/UI
 */
export function getErrorDisplayMessage(
  error: Error | string,
  t: (key: string) => string,
  fallbackMessage?: string,
  backendType?: string
): string {
  const parsedError = parseError(error, backendType)

  if (parsedError.type === 'generic_error') {
    // Show original error message for business errors (e.g., "Team not found")
    return parsedError.originalError || fallbackMessage || t('errors.generic_error')
  }

  // Use friendly message for specific error types
  return getUserFriendlyErrorMessage(error, t, backendType)
}
