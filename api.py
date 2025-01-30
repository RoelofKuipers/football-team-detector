from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Response
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uvicorn
from src.video_processor import process_football_video
from config import settings
import tempfile
import json


app = FastAPI(
    title="Football Video Analysis API",
    description="Upload football videos and get player detection results",
)

# Store ongoing processes
processing_jobs = {}


def process_video(video_path: str, job_id: str):
    """Background task to process the video"""
    try:
        # Create job-specific directory
        job_dir = settings.TEMP_BASE_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        processing_jobs[job_id]["status"] = "processing"

        # Process video
        results, output_video_path = process_football_video(
            video_path=video_path,
            output_dir=job_dir,
            model_path=settings.MODEL_PATH,
            cleanup_frames=True,
        )

        # Store results
        processing_jobs[job_id].update(
            {
                "status": "completed",
                "results": results,
                "output_video_path": str(output_video_path),
                "filename": Path(output_video_path).name,
            }
        )

        # Clean up the original uploaded video
        Path(video_path).unlink()

    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        if "job_dir" in processing_jobs[job_id]:
            shutil.rmtree(processing_jobs[job_id]["job_dir"])


@app.post("/upload-video")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a video file for processing
    """
    try:
        # Validate file type
        if not file.filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Only MP4 files are supported")
        # Create temporary file for video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path = temp_video.name

        # Save uploaded file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create job ID and start processing
        job_id = f"job_{len(processing_jobs) + 1}"
        processing_jobs[job_id] = {"status": "starting", "filename": file.filename}

        # Start processing in background
        background_tasks.add_task(process_video, temp_video_path, job_id)

        return {"message": "Video upload successful", "job_id": job_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Check the status of a video processing job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return processing_jobs[job_id]


@app.get("/jobs/{job_id}/results")
async def get_results(job_id: str):
    """
    Get the results of a completed video processing job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Results not ready. Current status: {job['status']}",
        )

    try:
        with open(job["results_path"], "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")


# TODO: We use storage in memory now, for S3 we will use a presigned link
@app.get("/jobs/{job_id}/video")
async def get_video(job_id: str):
    """Get processed video"""
    if job_id not in processing_jobs:
        return Response(status_code=404)

    job = processing_jobs[job_id]

    print("ROELOF")
    print(job["output_video_path"])
    print(job["filename"])

    if job["status"] in ["failed", "starting", "processing"]:
        return Response(status_code=400)

    if "output_video_path" not in job:
        return Response(status_code=404)

    video_path = Path(job["output_video_path"])
    if not video_path.exists() or video_path.stat().st_size == 0:
        return Response(status_code=404)

    try:
        return FileResponse(
            path=video_path,
            filename=job["filename"],
            media_type="video/mp4",
        )
    except Exception as e:
        raise Response(status_code=500)


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
