import os
import re
import json
import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any

import azure.functions as func

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your services
try:
    from services.azure_blob_service import AzureBlobService
    from services.document_parser import DocumentParser
    from services.ai_screening_service import AIScreeningService
    from services.cosmos_db_service import CosmosDBService
    from config import settings
    logging.info("Successfully imported all service modules")
except ImportError as e:
    logging.error(f"Failed to import modules: {str(e)}")
    logging.error(f"Python path: {sys.path}")
    raise

# Create FunctionApp
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def parse_event_grid_message(message_body: Any) -> Dict[str, str]:
    """Parse Event Grid message to extract job_id, blob_url, and filename"""
    try:
        if isinstance(message_body, list):
            event = message_body[0]
        else:
            event = message_body

        blob_url = event.get("data", {}).get("url")
        if not blob_url:
            raise ValueError("No blob URL found in event data")

        subject = event.get("subject", "")
        match = re.search(r'/containers/([^/]+)/blobs/(.+)$', subject)
        if not match:
            raise ValueError(f"Cannot parse blob path from subject: {subject}")

        container_name = match.group(1)
        blob_path = match.group(2)

        path_parts = blob_path.split("/", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Invalid blob path format: {blob_path}")

        job_id = path_parts[0]
        filename = path_parts[1]

        logging.info(f"Parsed: Container={container_name}, Job={job_id}, File={filename}")

        return {
            "job_id": job_id,
            "resume_blob_url": blob_url,
            "resume_filename": filename,
            "container_name": container_name,
        }

    except Exception as e:
        logging.error(f"Error parsing Event Grid message: {str(e)}")
        logging.debug(f"Message body: {json.dumps(message_body, default=str)[:2000]}")
        raise


async def process_resume(message_data: Dict[str, str]) -> None:
    """
    Process a single resume screening message
    ✅ UPDATED: Added duplicate check and removed total_resumes increment
    """
    blob_service = AzureBlobService()
    document_parser = DocumentParser()
    ai_service = AIScreeningService()
    cosmos_service = CosmosDBService()

    job_id = message_data["job_id"]
    resume_blob_url = message_data["resume_blob_url"]
    resume_filename = message_data["resume_filename"]

    logging.info(f"\n{'='*60}")
    logging.info(f"🔄 Processing Resume")
    logging.info(f"{'='*60}")
    logging.info(f"   📄 Filename: {resume_filename}")
    logging.info(f"   🆔 Job ID: {job_id}")

    try:
        # ✅ STEP 0: Check if already processed (DUPLICATE CHECK)
        is_duplicate = await cosmos_service.is_resume_already_processed(job_id, resume_filename)
        if is_duplicate:
            logging.info(f"   ⚠️  Resume already processed - skipping duplicate")
            logging.info(f"{'='*60}\n")
            return

        # 1. Get job description
        logging.info(f"\n   📚 Step 1: Fetching job description...")
        
        query = "SELECT * FROM c WHERE c.job_id = @job_id"
        parameters = [{"name": "@job_id", "value": job_id}]

        jobs = list(
            cosmos_service.jobs_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )

        if not jobs:
            raise Exception(f"Job description not found for job_id: {job_id}")

        job_data = jobs[0]
        user_id = job_data["user_id"]

        logging.info(f"      ✅ Job found: {job_data.get('screening_name')}")
        logging.info(f"      👤 User ID: {user_id}")

        # 2. Initialize screening job tracker (if first resume)
        logging.info(f"\n   📊 Step 2: Updating screening job tracker...")
        
        screening_job = await cosmos_service.get_screening_job_by_job_id(job_id)
        if not screening_job:
            logging.info(f"      🆕 Creating new screening job tracker")
            await cosmos_service.initialize_screening_job_for_job(job_id, user_id)

        # ✅ REMOVED: Don't increment total_resumes - we count from blob storage

        # 3. Download resume from blob
        logging.info(f"\n   ⬇️  Step 3: Downloading resume from blob storage...")
        resume_content = await blob_service.download_file(resume_blob_url)
        file_size_mb = len(resume_content) / (1024 * 1024)
        logging.info(f"      ✅ Downloaded: {file_size_mb:.2f} MB")

        # 4. Parse resume to text
        logging.info(f"\n   📄 Step 4: Parsing resume text...")
        resume_text = await document_parser.parse_document(resume_content, resume_filename)
        text_length = len(resume_text)
        logging.info(f"      ✅ Extracted: {text_length} characters")

        # 5. AI screening
        logging.info(f"\n   🤖 Step 5: Performing AI screening...")
        logging.info(f"      ⏳ This may take 30-60 seconds...")
        
        screening_result = await ai_service.screen_candidate(
            resume_text=resume_text,
            job_description=job_data["job_description_text"],
            must_have_skills=job_data["must_have_skills"],
            nice_to_have_skills=job_data["nice_to_have_skills"],
        )
        
        logging.info(f"      ✅ AI screening completed")
        logging.info(f"      📊 Fit Score: {screening_result['fit_score']['score']}%")
        logging.info(f"      👤 Candidate: {screening_result['candidate_info']['name']}")

        # 6. Build candidate report (with defensive checks)
        logging.info(f"\n   📝 Step 6: Creating candidate report...")
        
        from models import CandidateReport

        # Validate AI summary
        ai_summary = screening_result["ai_summary"]
        if not ai_summary or len(ai_summary) < 3:
            ai_summary = [
                f"Candidate with {screening_result['candidate_info']['total_experience']} of experience",
                f"Position: {screening_result['candidate_info']['position']}",
                f"Skills match: {screening_result['skills_analysis']['must_have_matched']}/{screening_result['skills_analysis']['must_have_total']}"
            ]

        # Validate career gap
        professional_summary = screening_result["professional_summary"].copy()
        career_gap = professional_summary.get("career_gap")
        if career_gap and (not career_gap.get("duration") or not isinstance(career_gap.get("duration"), str)):
            professional_summary["career_gap"] = None

        candidate_report = CandidateReport(
            candidate_name=screening_result["candidate_info"]["name"],
            email=screening_result["candidate_info"].get("email"),
            phone=screening_result["candidate_info"].get("phone"),
            position=screening_result["candidate_info"]["position"],
            location=screening_result["candidate_info"]["location"],
            total_experience=screening_result["candidate_info"]["total_experience"],
            resume_url=resume_blob_url,
            resume_filename=resume_filename,
            fit_score=screening_result["fit_score"],
            must_have_skills_matched=screening_result["skills_analysis"]["must_have_matched"],
            must_have_skills_total=screening_result["skills_analysis"]["must_have_total"],
            nice_to_have_skills_matched=screening_result["skills_analysis"]["nice_to_have_matched"],
            nice_to_have_skills_total=screening_result["skills_analysis"]["nice_to_have_total"],
            matched_must_have_skills=screening_result["skills_analysis"]["matched_must_have_list"],
            matched_nice_to_have_skills=screening_result["skills_analysis"]["matched_nice_to_have_list"],
            ai_summary=ai_summary,
            skill_depth_analysis=screening_result["skill_depth_analysis"],
            professional_summary=professional_summary,
            company_tier_analysis=screening_result["company_tier_analysis"],
        )
        
        logging.info(f"      ✅ Candidate report created")

        # 7. Save screening result
        logging.info(f"\n   💾 Step 7: Saving screening result to database...")
        
        screening_id = await cosmos_service.save_screening_result(
            job_id=job_id, user_id=user_id, candidate_report=candidate_report.dict()
        )
        
        logging.info(f"      ✅ Saved with ID: {screening_id}")

        # 8. Update screening progress
        logging.info(f"\n   📈 Step 8: Updating progress tracker...")
        
        await cosmos_service.update_screening_job_progress_by_job_id(
            job_id=job_id, resume_filename=resume_filename, status="success", screening_id=screening_id
        )

        logging.info(f"\n{'='*60}")
        logging.info(f"✅ SUCCESS!")
        logging.info(f"{'='*60}")
        logging.info(f"   Candidate: {screening_result['candidate_info']['name']}")
        logging.info(f"   Fit Score: {screening_result['fit_score']['score']}%")
        logging.info(f"   Screening ID: {screening_id}")
        logging.info(f"{'='*60}\n")

    except Exception as exc:
        error_msg = str(exc)
        error_trace = traceback.format_exc()
        
        logging.error(f"\n{'='*60}")
        logging.error(f"❌ PROCESSING FAILED")
        logging.error(f"{'='*60}")
        logging.error(f"   Error: {error_msg}")
        logging.error(f"   Filename: {resume_filename}")
        logging.error(f"   Job ID: {job_id}")
        logging.error(f"{'='*60}\n")
        logging.debug(error_trace)
        
        # Attempt to mark the screening as failed in the tracker
        try:
            cosmos_service_fallback = CosmosDBService()
            await cosmos_service_fallback.update_screening_job_progress_by_job_id(
                job_id=job_id, resume_filename=resume_filename, status="failed"
            )
        except Exception as update_exc:
            logging.debug(f"Could not update failure status: {str(update_exc)}")
        
        # Re-raise with serializable error message
        raise Exception(f"Resume processing failed: {error_msg}")


@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name="resume-processing-queue",
    connection="AZURE_SERVICE_BUS_CONNECTION_STRING"
)
async def resume_processor(msg: func.ServiceBusMessage) -> None:
    """
    Azure Function triggered by Service Bus queue message
    Processes resume screening jobs from the queue with enhanced logging
    """
    logging.info("📬 Service Bus queue trigger function processing message")

    try:
        # Parse message body (Event Grid payload)
        body_bytes = msg.get_body()
        try:
            message_body_str = body_bytes.decode("utf-8")
        except Exception:
            # msg.get_body() sometimes returns already-decoded str
            message_body_str = str(body_bytes)

        logging.debug(f"Message body (truncated): {message_body_str[:1000]}")
        message_body = json.loads(message_body_str)

        # Convert Event Grid data to our message_data format
        message_data = parse_event_grid_message(message_body)

        # Await async processing function
        await process_resume(message_data)

        logging.info("✅ Message processed successfully")

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logging.error(f"❌ Error processing message: {error_msg}")
        logging.debug(error_trace)
        
        # Re-raise with serializable error message
        raise Exception(f"Message processing failed: {error_msg}")
