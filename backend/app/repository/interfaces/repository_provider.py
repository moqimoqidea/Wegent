# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Repository provider interface, defining methods related to code repositories
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from app.models.user import User


class RepositoryProvider(ABC):
    """
    Repository provider interface, defining methods related to code repositories
    Different code repository services (GitHub, GitLab, etc.) need to implement this interface
    """
    @abstractmethod
    async def get_repositories(
        self,
        user: User,
        page: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get the user's repository list
        
        Args:
            user: User object
            page: Page number
            limit: Number per page
            
        Returns:
            Repository list
            
        Raises:
            HTTPException: Exception thrown when fetching fails
        """
    
    @abstractmethod
    async def get_branches(
        self,
        user: User,
        repo_name: str,
        git_domain: str
    ) -> List[Dict[str, Any]]:
        """
        Get the branch list of the specified repository
        
        Args:
            user: User object
            repo_name: Repository name
            
        Returns:
            Branch list
            
        Raises:
            HTTPException: Exception thrown when fetching fails
        """
    
    @abstractmethod
    def validate_token(
        self,
        token: str
    ) -> Dict[str, Any]:
        """
        Validate code repository token
        
        Args:
            token: Code repository token
            
        Returns:
            Validation result, including whether it's valid, user information, etc.
            
        Raises:
            HTTPException: Exception thrown when validation fails
        """
    
    @abstractmethod
    async def search_repositories(
        self,
        user: User,
        query: str,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search the user's code repositories
        
        Args:
            user: User object
            query: Search keyword
            timeout: Timeout (seconds)
            
        Returns:
            Search results
            
        Raises:
            HTTPException: Exception thrown when search fails
        """
