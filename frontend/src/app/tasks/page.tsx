// SPDX-FileCopyrightText: 2025 Weibo, Inc.
//
// SPDX-License-Identifier: Apache-2.0

'use client'

import { Suspense, useState } from 'react'
import { UserProvider } from '@/features/common/UserContext'
import { TaskContextProvider } from '@/features/tasks/contexts/taskContext'
import { teamService } from '@/features/tasks/service/teamService'
import TopNavigation from '@/features/layout/TopNavigation'
import UserMenu from '@/features/layout/UserMenu'
import TaskSidebar from '@/features/tasks/components/TaskSidebar'
import BeginnerGuideModal from '@/features/tasks/components/BeginnerGuideModal'
import ChatArea from '@/features/tasks/components/ChatArea'
import TaskParamSync from '@/features/tasks/components/TaskParamSync'
import TeamShareHandler from '@/features/tasks/components/TeamShareHandler'
import OidcTokenHandler from '@/features/login/components/OidcTokenHandler'
import '@/app/tasks/tasks.css'
import '@/features/common/scrollbar.css'
import { GithubStarButton } from '@/features/layout/GithubStarButton'
import { Team } from '@/types/api'
export default function TasksPage() {
  // Team state from service
  const { teams, isTeamsLoading, refreshTeams } = teamService.useTeams()
  
  // Mobile sidebar state
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  
  // Selected team state for sharing
  const [selectedTeamForNewTask, setSelectedTeamForNewTask] = useState<Team | null>(null)

  const handleRefreshTeams = async (): Promise<Team[]> => {
    return await refreshTeams()
  }

  return (
    <UserProvider>
      <TaskContextProvider>
        {/* Handle OIDC token from URL parameters */}
        <OidcTokenHandler />
        <Suspense>
          <TaskParamSync />
        </Suspense>
        <Suspense>
          <TeamShareHandler
            teams={teams}
            onTeamSelected={setSelectedTeamForNewTask}
            onRefreshTeams={handleRefreshTeams}
          />
        </Suspense>
          {/* Beginner guide modal */}
          <BeginnerGuideModal
            teams={teams}
            teamLoading={isTeamsLoading}
          />
          <div className="flex smart-h-screen bg-base text-text-primary box-border">
            {/* Responsive sidebar */}
            <TaskSidebar
              isMobileSidebarOpen={isMobileSidebarOpen}
              setIsMobileSidebarOpen={setIsMobileSidebarOpen}
            />
            {/* Main content area */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Top navigation */}
              <TopNavigation
                activePage="tasks"
                showLogo={false}
                onMobileSidebarToggle={() => setIsMobileSidebarOpen(true)}
              >
                <GithubStarButton />
                <UserMenu />
              </TopNavigation>
              {/* Chat area */}
              <ChatArea
                teams={teams}
                isTeamsLoading={isTeamsLoading}
                selectedTeamForNewTask={selectedTeamForNewTask}
              />
            </div>
          </div>
      </TaskContextProvider>
    </UserProvider>
  )
}
