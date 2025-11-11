from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import gc
import threading
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("Starting face model preload...")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, get_face_app)
        print("✓ Face model preloaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to preload face model at startup: {e}")
        print("The model will be downloaded on first request (this may take 5-10 minutes)")

    yield

    # Shutdown (if needed)
    print("Shutting down...")


api = FastAPI(title="Face Recognition API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware for Next.js
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
face_app = None
mongo_client = None
db = None
users_collection = None
face_app_lock = threading.Lock()
face_app_ready = False


def get_mongo_connection():
    """Lazy load MongoDB connection - connects to YOUR app's database"""
    global mongo_client, db, users_collection
    if mongo_client is None:
        from pymongo import MongoClient
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("MONGO_URI environment variable not set")

        mongo_client = MongoClient(MONGO_URI)

        # Use the SAME database as your Next.js app (not face_attendance)
        # Extract database name from connection string or use default
        db_name = os.getenv("MONGO_DB_NAME", "test")  # Change "test" to your actual DB name
        db = mongo_client[db_name]

        # Use the SAME collection as Better Auth (singular 'user', not 'users')
        users_collection = db["user"]

    return users_collection


def get_face_app():
    """Thread-safe lazy initializer for the insightface model"""
    global face_app, face_app_ready
    if face_app is None:
        with face_app_lock:
            if face_app is None:
                try:
                    from insightface.app import FaceAnalysis
                    _app = FaceAnalysis(providers=['CPUExecutionProvider'])
                    _app.prepare(ctx_id=0, det_size=(640, 640))
                    face_app = _app
                    face_app_ready = True
                except Exception as e:
                    face_app = None
                    face_app_ready = False
                    raise RuntimeError(f"Failed to initialize face model: {str(e)}")
    if not face_app_ready:
        raise RuntimeError("Face model not ready")
    return face_app


def ensure_face_app_or_503():
    """Helper to return 503 if model is not ready yet"""
    try:
        return get_face_app()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Face model is not ready yet. The model is downloading (this can take 5-10 minutes on first run). Please try again later. Error: {str(e)}"
        )


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@api.get("/")
async def root():
    """Root endpoint with API status"""
    try:
        users_collection = get_mongo_connection()
        registered_count = users_collection.count_documents({"faceRegistered": True})

        return {
            "message": "Face Recognition API is running",
            "version": "1.0.0",
            "features": ["user_registration", "face_recognition", "mongodb_storage"],
            "registered_users": registered_count,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "message": "Face Recognition API is running",
            "version": "1.0.0",
            "status": "healthy",
            "database_status": "connection_pending"
        }


@api.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        users_collection = get_mongo_connection()
        # Test database connection
        users_collection.find_one()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "memory_optimized": True
    }


@api.post("/register/")
async def register_face(user_id: str = Form(...), file: UploadFile = File(...)):
    """Register a user's face - COMPATIBLE WITH BETTER AUTH"""
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face analysis model and MongoDB connection
        app_insight = ensure_face_app_or_503()
        users_collection = get_mongo_connection()

        # Detect faces
        faces = app_insight.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        if len(faces) > 1:
            raise HTTPException(status_code=400,
                                detail="Multiple faces detected. Please upload an image with only one face")

        # Get face embedding
        face = faces[0]
        emb = face.embedding

        # Find user by email (Better Auth uses email as identifier)
        existing_user = users_collection.find_one({"email": user_id})

        if not existing_user:
            raise HTTPException(status_code=404, detail=f"User with email {user_id} not found")

        # Save embedding to MongoDB (same collection as Better Auth)
        result = users_collection.update_one(
            {"email": user_id},
            {
                "$set": {
                    "faceEmbedding": emb.tolist(),
                    "faceRegistered": True,
                    "faceRegisteredAt": datetime.utcnow(),
                    "faceDetectionConfidence": float(face.det_score),
                    "faceAge": int(face.age) if hasattr(face, 'age') else None,
                    "faceGender": face.sex if hasattr(face, 'sex') else None,
                    "updatedAt": datetime.utcnow()
                }
            }
        )

        # Force garbage collection
        gc.collect()

        return JSONResponse({
            "message": f"Face registered successfully for user {user_id}",
            "user_id": user_id,
            "action": "updated",
            "face_quality": {
                "detection_confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.sex if hasattr(face, 'sex') else None
            },
            "database_operation": "success"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@api.post("/recognize/")
async def recognize_face(file: UploadFile = File(...), threshold: float = 0.6):
    """Recognize a user's face - COMPATIBLE WITH BETTER AUTH"""
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face analysis model and MongoDB connection
        app_insight = ensure_face_app_or_503()
        users_collection = get_mongo_connection()

        # Detect faces
        faces = app_insight.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        face = faces[0]
        emb = face.embedding

        # Compare against all embeddings in MongoDB (Better Auth collection)
        best_score = 0
        best_user_email = None
        best_user_data = None

        registered_users = users_collection.find({
            "faceRegistered": True,
            "faceEmbedding": {"$exists": True, "$ne": []}
        })

        comparison_count = 0
        for user in registered_users:
            comparison_count += 1
            db_emb = np.array(user["faceEmbedding"])
            score = cosine_similarity(emb, db_emb)

            if score > best_score:
                best_score = score
                best_user_email = user["email"]
                best_user_data = user

        # Force garbage collection
        gc.collect()

        # Determine recognition result
        is_recognized = best_user_email is not None and best_score > threshold

        result = {
            "recognized": is_recognized,
            "user_id": best_user_email if is_recognized else None,
            "user_email": best_user_email if is_recognized else None,
            "confidence_score": float(best_score),
            "threshold_used": threshold,
            "detection_quality": {
                "detection_confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.sex if hasattr(face, 'sex') else None
            },
            "comparison_stats": {
                "users_compared": comparison_count,
                "best_match_score": float(best_score)
            }
        }

        # Add user details if recognized
        if is_recognized and best_user_data:
            result["user_details"] = {
                "name": best_user_data.get("name"),
                "email": best_user_data.get("email"),
                "registered_age": best_user_data.get("faceAge"),
                "registered_gender": best_user_data.get("faceGender"),
                "registration_confidence": best_user_data.get("faceDetectionConfidence"),
                "registered_at": str(best_user_data.get("faceRegisteredAt"))
            }

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@api.get("/users/")
async def get_registered_users():
    """Get list of all registered users - COMPATIBLE WITH BETTER AUTH"""
    try:
        users_collection = get_mongo_connection()

        users = []
        for user in users_collection.find(
                {"faceRegistered": True},
                {"email": 1, "name": 1, "faceAge": 1, "faceGender": 1, "faceDetectionConfidence": 1,
                 "faceRegisteredAt": 1, "_id": 0}
        ):
            users.append(user)

        return JSONResponse({
            "total_registered_users": len(users),
            "users": users
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")


@api.delete("/users/{user_email}")
async def delete_user_face(user_email: str):
    """Delete a specific user's face registration - COMPATIBLE WITH BETTER AUTH"""
    try:
        users_collection = get_mongo_connection()

        result = users_collection.update_one(
            {"email": user_email},
            {
                "$unset": {
                    "faceEmbedding": "",
                    "faceRegistered": "",
                    "faceRegisteredAt": "",
                    "faceDetectionConfidence": "",
                    "faceAge": "",
                    "faceGender": ""
                },
                "$set": {
                    "updatedAt": datetime.utcnow()
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"User {user_email} not found")

        return JSONResponse({
            "message": f"User {user_email} face registration deleted successfully",
            "user_email": user_email
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user face: {str(e)}")


@api.get("/stats/")
async def get_system_stats():
    """Get system statistics - COMPATIBLE WITH BETTER AUTH"""
    try:
        users_collection = get_mongo_connection()

        total_users = users_collection.count_documents({})
        registered_faces = users_collection.count_documents({"faceRegistered": True})

        return JSONResponse({
            "total_users": total_users,
            "registered_faces": registered_faces,
            "registration_rate": f"{(registered_faces / total_users * 100):.1f}%" if total_users > 0 else "0%"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@api.get("/debug/database-info")
async def get_database_info():
    """Diagnostic endpoint to check database configuration and users"""
    try:
        from pymongo import MongoClient

        MONGO_URI = os.getenv("MONGO_URI")
        MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "test")

        if not MONGO_URI:
            return JSONResponse({
                "error": "MONGO_URI not set in environment variables",
                "env_file_path": str(env_path),
                "env_file_exists": env_path.exists()
            })

        client = MongoClient(MONGO_URI)

        # List all databases
        all_databases = client.list_database_names()

        # Get current database
        db = client[MONGO_DB_NAME]

        # List collections in current database
        collections = db.list_collection_names()

        # Get users from 'user' collection
        users_collection = db["user"]
        total_users = users_collection.count_documents({})

        # Get sample user emails (first 5)
        sample_users = []
        for user in users_collection.find({}, {"email": 1, "name": 1, "_id": 0}).limit(5):
            sample_users.append(user)

        return JSONResponse({
            "current_config": {
                "database_name": MONGO_DB_NAME,
                "collection_name": "user",
                "mongo_uri_set": bool(MONGO_URI),
                "env_file_loaded": env_path.exists()
            },
            "available_databases": all_databases,
            "collections_in_current_db": collections,
            "user_collection_stats": {
                "total_users": total_users,
                "sample_users": sample_users
            },
            "instructions": "If total_users is 0, change MONGO_DB_NAME in your .env file to one of the available_databases"
        })

    except Exception as e:
        import traceback
        return JSONResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status_code=500)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
