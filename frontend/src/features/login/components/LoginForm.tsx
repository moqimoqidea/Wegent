// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

'use client'

import { useEffect, useState } from 'react'
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline'
import { Button } from 'antd'
import { useRouter } from 'next/navigation'
import { useUser } from '@/features/common/UserContext'
import { paths } from '@/config/paths'
import { App } from 'antd'
import { useTranslation } from '@/hooks/useTranslation'
import LanguageSwitcher from '@/components/LanguageSwitcher'
import { ThemeToggle } from '@/features/theme/ThemeToggle'

export default function LoginForm() {
  const { t } = useTranslation('common')
  const { message } = App.useApp()
  const router = useRouter()
  const [formData, setFormData] = useState({
    user_name: 'admin',
    password: 'Wegent2025!'
  })
  const [showPassword, setShowPassword] = useState(false)
      // Used antd message.error for unified error prompt, no need for local error state
  const [isLoading, setIsLoading] = useState(false)

      // Get login mode configuration
  const loginMode = process.env.NEXT_PUBLIC_LOGIN_MODE || 'all'
  const showPasswordLogin = loginMode === 'password' || loginMode === 'all'
  const showOidcLogin = loginMode === 'oidc' || loginMode === 'all'

      // Get OIDC login button text
  const oidcLoginText = process.env.NEXT_PUBLIC_OIDC_LOGIN_TEXT || t('login.oidc_login')

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
        // Used antd message.error for unified error prompt, no need for local error state
  }

  const { user, refresh, isLoading: userLoading, login } = useUser()

  useEffect(() => {
    if (!userLoading && user) {
      router.replace(paths.task.getHref())
    }
  }, [userLoading, user, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (user) {
      router.replace(paths.task.getHref())
      return
    }
    setIsLoading(true)
        // Used antd message.error for unified error prompt, no need for local error state

    try {
      await login({
        user_name: formData.user_name,
        password: formData.password
      })
      router.replace(paths.task.getHref())
    } catch (error: any) {
      // Error handling is already done in UserContext.login, no need to show error message here
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Language switcher */}
      <div className="absolute top-4 right-4 flex items-center gap-3">
        <ThemeToggle />
        <LanguageSwitcher />
      </div>
      {/* Password login form */}
      {showPasswordLogin && (
        <form className="space-y-6" onSubmit={handleSubmit}>
          <div>
            <label htmlFor="user_name" className="block text-sm font-medium text-text-secondary">
              {t('login.username')}
            </label>
            <div className="mt-1">
              <input
                id="user_name"
                name="user_name"
                type="text"
                autoComplete="username"
                required
                value={formData.user_name}
                onChange={handleInputChange}
                className="appearance-none block w-full px-3 py-2 border border-border rounded-md shadow-sm bg-base text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-transparent sm:text-sm"
                placeholder={t('login.enter_username')}
              />
            </div>
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-text-secondary">
              {t('login.password')}
            </label>
            <div className="mt-1 relative">
              <input
                id="password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                autoComplete="current-password"
                required
                value={formData.password}
                onChange={handleInputChange}
                className="appearance-none block w-full px-3 py-2 pr-10 border border-border rounded-md shadow-sm bg-base text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-transparent sm:text-sm"
                placeholder={t('login.enter_password')}
              />
               <button
                type="button"
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? (
                  <EyeIcon className="h-5 w-5 text-text-muted hover:text-text-secondary" />
                ) : (
                  <EyeSlashIcon className="h-5 w-5 text-text-muted hover:text-text-secondary" />
                )}
              </button>
            </div>
          </div>

          {/* Error prompts are unified with antd message, no longer rendered locally */}

          <div>
            <Button
              type="primary"
              htmlType="submit"
              disabled={isLoading}
              loading={isLoading}
              style={{ width: '100%' }}
            >
              {isLoading ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-primary-contrast" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {t('login.logging_in')}
                </div>
              ) : (
                t('user.login')
              )}
            </Button>
          </div>

          {/* Show test account info */}
          <div className="mt-6 text-center text-xs text-text-muted">
            {t('login.test_account')}
          </div>
        </form>
      )}

      {/* Divider and third-party login - only shown when both login modes are displayed */}
      {showPasswordLogin && showOidcLogin && (
        <div className="mt-6">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-border" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-surface text-text-muted">{t('login.or_continue_with')}</span>
            </div>
          </div>
        </div>
      )}

      {/* OIDC login */}
      {showOidcLogin && (
        <div className={showPasswordLogin ? "mt-6" : ""}>
          <div className="grid grid-cols-1 gap-3">
            <Button
              type="default"
              onClick={() => window.location.href = '/api/auth/oidc/login'}
              style={{ width: '100%', justifyContent: 'center', display: 'flex', alignItems: 'center' }}
              icon={<img src="/ocid.png" alt="OIDC Login" className="w-5 h-5 mr-2" />}
            >
              {oidcLoginText}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}
