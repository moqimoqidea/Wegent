# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.core import security
from app.models.user import User
from app.schemas.bot import BotCreate, BotUpdate, BotInDB, BotListResponse, BotDetail
from app.services.adapters import bot_kinds_service

router = APIRouter()

@router.get("", response_model=BotListResponse)
def list_bots(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(security.get_current_user)
):
    """Get current user's Bot list (paginated)"""
    skip = (page - 1) * limit
    bot_dicts = bot_kinds_service.get_user_bots(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit
    )
    if page == 1 and len(bot_dicts) < limit:
        total = len(bot_dicts)
    else:
        total = bot_kinds_service.count_user_bots(db=db, user_id=current_user.id)
    
    # bot_dicts are already in the correct format
    return {"total": total, "items": bot_dicts}

@router.post("", response_model=BotInDB, status_code=status.HTTP_201_CREATED)
def create_bot(
    bot_create: BotCreate,
    current_user: User = Depends(security.get_current_user),
    db: Session = Depends(get_db)
):
    """Create new Bot"""
    bot_dict = bot_kinds_service.create_with_user(db=db, obj_in=bot_create, user_id=current_user.id)
    return bot_dict

@router.get("/{bot_id}", response_model=BotDetail)
def get_bot(
    bot_id: int,
    current_user: User = Depends(security.get_current_user),
    db: Session = Depends(get_db)
):
    """Get specified Bot details with related user"""
    bot_dict = bot_kinds_service.get_bot_detail(db=db, bot_id=bot_id, user_id=current_user.id)
    return bot_dict

@router.put("/{bot_id}", response_model=BotInDB)
def update_bot(
    bot_id: int,
    bot_update: BotUpdate,
    current_user: User = Depends(security.get_current_user),
    db: Session = Depends(get_db)
):
    """Update Bot information"""
    bot_dict = bot_kinds_service.update_with_user(
        db=db,
        bot_id=bot_id,
        obj_in=bot_update,
        user_id=current_user.id
    )
    return bot_dict

@router.delete("/{bot_id}")
def delete_bot(
    bot_id: int,
    current_user: User = Depends(security.get_current_user),
    db: Session = Depends(get_db)
):
    """Delete Bot or deactivate if used in teams"""
    bot_kinds_service.delete_with_user(db=db, bot_id=bot_id, user_id=current_user.id)
    return {"message": "Bot deleted successfully"}