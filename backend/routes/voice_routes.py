from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from deps import get_current_employee
from speech import transcribe_audio, stt_available

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/available")
def available(emp: dict = Depends(get_current_employee)):
    return {"available": stt_available()}


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    emp: dict = Depends(get_current_employee),
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file.")
    text = transcribe_audio(audio_bytes)
    if not text:
        raise HTTPException(422, "Could not transcribe audio — please try again.")
    text = text.replace('"', "").replace("'", "").strip()
    return {"transcript": text}
