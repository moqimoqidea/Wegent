# SPDX-FileCopyrightText: 2025 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from urllib.parse import urlparse

from shared.logger import setup_logger

logger = setup_logger(__name__)


def get_repo_name_from_url(url):
    # Remove .git suffix if exists
    if url.endswith(".git"):
        url = url[:-4]  # Correctly remove '.git' suffix

    # Handle special path formats containing '/-/' (like tree structure or merge requests)
    if "/-/" in url:
        url = url.split("/-/")[0]

    parts = url.split("/")

    repo_name = parts[-1] if parts[-1] else parts[-2]
    return repo_name


def clone_repo(project_url, branch, project_path, user_name=None, token=None):
    """
    Clone repository to specified path

    Returns:
        Tuple (success, message):
        - On success: (True, None)
        - On failure: (False, error_message)
    """
    if  not token or token == "***":
        token = get_git_token_from_url(project_url)
    if user_name is None:
        user_name = "token"
    logger.info(
        f"get git token from url: {project_url}, branch:{branch}, project:{project_path}"
    )
    if token:
        return clone_repo_with_token(
            project_url, branch, project_path, user_name, token
        )
    return False, "Token is not provided"


def get_domain_from_url(url):
    if "/-/" in url:
        url = url.split("/-/")[0]

    # Handle SSH format (ssh://git@domain.com:port/...)
    if url.startswith("ssh://"):
        url = url[6:]  # Remove ssh:// prefix

    # Handle git@domain.com: format
    if "@" in url and ":" in url:
        # Extract domain:port part
        return url.split("@")[1].split(":")[0]

    # Parse standard URL using urlparse
    parsed = urlparse("https://" + url if "://" not in url else url)

    return parsed.hostname if parsed.netloc else ""


def clone_repo_with_token(project_url, branch, project_path, username, token):

    if project_url.startswith("https://"):
        username = "token"
        protocol, rest = project_url.split("://", 1)
        auth_url = f"{protocol}://{username}:{token}@{rest}"
    else:
        auth_url = project_url

    # Build basic command
    cmd = ["git", "clone"]

    # Add branch parameter if branch is specified
    if branch:
        cmd.extend(["--branch", branch, "--single-branch"])

    # Add URL and path
    cmd.extend([auth_url, project_path])
    try:
        # Use subprocess.run to capture output and errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(
            f"git clone url: {project_url}, cmd: {cmd}, code: {result.returncode}"
        )
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"git clone failed: {error_msg}")
        return False, error_msg
    except Exception as e:
        logger.error(f"git clone failed with unexpected error: {e}")
        return False, str(e)


def get_git_token_from_url(git_url):
    domain = get_domain_from_url(git_url)
    if not domain:
        logger.error(f"get domain from url failed: {git_url}")
        raise Exception(f"get domain from url failed: {git_url}")

    token_file = f"/root/.ssh/{domain}"
    try:
        with open(token_file, "r") as f:
            return f.read().strip()
    except IOError:
        raise Exception(f"get domain from file failed: {git_url}, file: {token_file}")


def get_project_path_from_url(url):

    # Handle special path formats containing '/-/'
    if "/-/" in url:
        url = url.split("/-/")[0]

    # Remove .git suffix if exists
    if url.endswith(".git"):
        url = url[:-4]

    # Handle SSH format (git@domain.com:user/repo)
    if "@" in url and ":" in url:
        # Extract user/repo part
        return url.split(":")[-1]

    # Parse standard URL using urlparse
    parsed = urlparse("https://" + url if "://" not in url else url)

    # Remove leading slash
    path = parsed.path
    if path.startswith("/"):
        path = path[1:]

    return path


def set_git_config(repo_path, name, email):
    """
    Set git config user.name and user.email for a repository
    
    Args:
        repo_path: Path to the git repository
        name: Git user name to set
        email: Git user email to set
        
    Returns:
        Tuple (success, message):
        - On success: (True, None)
        - On failure: (False, error_message)
    """
    try:
        # Set both user.name and user.email in a single command
        cmd = f"git config user.name \"{name}\" && git config user.email \"{email}\""
        result = subprocess.run(cmd, cwd=repo_path, shell=True, capture_output=True, text=True, check=True)
        
        logger.info(f"Git config set successfully in {repo_path}: user.name={name}, user.email={email}")
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"Failed to set git config: {error_msg}")
        return False, error_msg
    except Exception as e:
        logger.error(f"Failed to set git config with unexpected error: {e}")
        return False, str(e)
