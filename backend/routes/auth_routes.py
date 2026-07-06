from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import auth
from deps import get_current_employee

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(body: LoginBody):
    emp = auth.fetch_by_email(body.email)
    if not emp or not auth.verify_password(body.password, emp["password_hash"]):
        raise HTTPException(401, "Invalid credentials.")
    token = auth.make_token(emp["id"])
    # Token goes in the response body — the frontend stores it and sends it
    # back as `Authorization: Bearer <token>` on every subsequent request.
    # (Previously this was an httpOnly cookie, but Hugging Face Spaces'
    # shared proxy was stripping the Access-Control-Allow-Credentials header
    # that cross-origin cookies require, so this avoids that dependency.)
    return {"employee": auth.public_employee(emp), "token": token}


@router.post("/logout")
def logout():
    # Nothing to do server-side — the token is stateless (HMAC-signed with
    # an expiry, same scheme as before). The frontend just discards it.
    return {"ok": True}


@router.get("/me")
def me(emp: dict = Depends(get_current_employee)):
    return {"employee": auth.public_employee(emp)}