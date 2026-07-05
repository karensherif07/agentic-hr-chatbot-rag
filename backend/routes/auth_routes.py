from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

import auth
from deps import get_current_employee, COOKIE_NAME

router = APIRouter(prefix="/api/auth", tags=["auth"])

_COOKIE_KWARGS = dict(
    httponly=True,
    samesite="lax",
    secure=False,   # set True in production behind HTTPS
    max_age=auth.SESSION_TTL_SEC,
    path="/",
)


class LoginBody(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(body: LoginBody, response: Response):
    emp = auth.fetch_by_email(body.email)
    if not emp or not auth.verify_password(body.password, emp["password_hash"]):
        raise HTTPException(401, "Invalid credentials.")
    token = auth.make_token(emp["id"])
    response.set_cookie(COOKIE_NAME, token, **_COOKIE_KWARGS)
    return {"employee": auth.public_employee(emp)}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(emp: dict = Depends(get_current_employee)):
    return {"employee": auth.public_employee(emp)}
