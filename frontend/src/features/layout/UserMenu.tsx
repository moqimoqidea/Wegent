// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

'use client'

import { Menu } from '@headlessui/react'
import { Button } from 'antd'

import { useUser } from '@/features/common/UserContext'
import { useTranslation } from '@/hooks/useTranslation'
import { DocsButton } from '@/features/layout/DocsButton'
import { ThemeToggle } from '@/features/theme/ThemeToggle'

type UserMenuProps = {
  className?: string
}

export default function UserMenu({ className = '' }: UserMenuProps) {
  const { t } = useTranslation('common')
  const { user, logout } = useUser()
  const userDisplayName = user?.user_name || t('user.default_name')

  return (
    <div className={className}>
      <Menu as="div" className="relative">
        <Menu.Button className="px-4 py-1.5 bg-muted border border-border rounded-full flex items-center justify-center text-base font-semibold text-text-primary hover:bg-border/40 transition-colors duration-200">
          {userDisplayName}
        </Menu.Button>
        <Menu.Items
          className="absolute top-full right-0 mt-2 min-w-[120px] rounded-lg border border-border bg-surface py-1 z-30 focus:outline-none"
          style={{ boxShadow: 'var(--shadow-popover)' }}
        >
          <div className="flex flex-col gap-2 px-2 pb-1.5">
            <DocsButton className="w-full justify-center" />
            <ThemeToggle className="w-full justify-center" />
          </div>
          <div className="my-1 h-px bg-border/60" />
          <Menu.Item>
            {({ active }) => (
              <Button
                type="text"
                onClick={logout}
                className={`!w-full !text-left !px-2 !py-1.5 !text-xs !text-text-primary ${
                  active ? '!bg-muted' : '!bg-transparent'
                }`}
              >
                {t('user.logout')}
              </Button>
            )}
          </Menu.Item>
        </Menu.Items>
      </Menu>
    </div>
  )
}
