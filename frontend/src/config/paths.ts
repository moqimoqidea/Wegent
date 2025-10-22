// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

export const paths = {
    home: {
        getHref: () => '/',
    },
    docs: {
        getHref: () => process.env.NEXT_PUBLIC_DOCS_URL || 'https://github.com/wecode-ai/Wegent',
    },
    auth: {
        password_login: {
            getHref: () => '/login',
        },
        login: {
            getHref: () => {
                console.log(typeof window === 'undefined')
                // Always return local login page in SSR/Node environment
                return paths.auth.password_login.getHref()
            },
        },
    },
    task: {
        getHref: () => '/tasks',
    },
    settings: {
        root: {
            getHref: () => '/settings',
        },
        integrations: {
            getHref: () => '/settings',
        },
        bot: {
            getHref: () => '/settings?tab=bot',
        },
        team: {
            getHref: () => '/settings?tab=team',
        },
    },
} as const;
