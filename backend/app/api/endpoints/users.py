# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.core import security
from app.api.dependencies import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate, UserInDB
from app.services.user import user_service

router = APIRouter()

@router.get("/me", response_model=UserInDB)
async def read_current_user(
    current_user: User = Depends(security.get_current_user)
):
    """Get current user information"""
    return current_user

@router.put("/me", response_model=UserInDB)
async def update_current_user_endpoint(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """Update current user information"""
    try:
        user = user_service.update_current_user(
            db=db,
            user=current_user,
            obj_in=user_update,
        )
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
def create_user(
    user_create: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create new user"""
    return user_service.create_user(db=db, obj_in=user_create, background_tasks=background_tasks)