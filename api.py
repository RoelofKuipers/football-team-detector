from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uvicorn
from src.video_processor import process_football_video
import tempfile
import json

app = FastAPI(
    title="Football Video Analysis API",
    description="Upload football videos and get player detection results",
)

# Store ongoing processes
processing_jobs = {}

# Initialize the model
MODEL_PATH = "checkpoints/yolo_football.pt"


def process_video(video_path: str, job_id: str):
    """Background task to process the video"""
    try:
        # Create temporary directories
        temp_dir = Path(tempfile.mkdtemp())
        # Update job status
        processing_jobs[job_id]["status"] = "processing"

        # Process video using shared function
        results, _ = process_football_video(
            video_path=video_path,
            output_dir=temp_dir,
            model_path=MODEL_PATH,
            cleanup_frames=True,
        )

        # Save results
        results_path = Path("output") / f"{job_id}_results.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f)

        # Update job status
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["results_path"] = str(results_path)

        # Cleanup
        shutil.rmtree(temp_dir)

    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)


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


# TODO: For local storage, for S3 we will use a presigned link
@app.get("/jobs/{job_id}/video")
async def get_video(job_id: str):
    """Get processed video"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not complete")

    video_path = Path("output") / f"match_processed.mp4"
    print(video_path)
    if not video_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Video file not found {video_path}"
        )

    return FileResponse(
        path=video_path, filename=f"processed_{job['filename']}", media_type="video/mp4"
    )


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


# import requests

# # Upload video
# with open('path/to/video.mp4', 'rb') as f:
#     response = requests.post(
#         'http://localhost:8000/upload-video',
#         files={'file': f}
#     )
# job_id = response.json()['job_id']

# # Check status
# status = requests.get(f'http://localhost:8000/status/{job_id}')
# print(status.json())

# # Get results
# results = requests.get(f'http://localhost:8000/results/{job_id}')
# print(results.json())
