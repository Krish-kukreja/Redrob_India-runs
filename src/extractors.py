import datetime
import re
import spacy

# Load the NLP model once at the module level so it doesn't reload for every candidate
nlp = spacy.load("en_core_web_sm")

class CandidateExtractor:
    """
    This is the class which has all the methods to extract the needed data from the jsons of the candidates 
    """
    
    def __init__(self, reference_date_str: str = "2026-06-15" ):
        self.ref_date = datetime.datetime.strptime(reference_date_str, '%Y-%m-%d').date()
        
    def _smart_truncate(self, text: str, min_chars: int) -> str:
        """
        Reads up to 'min_chars', then looks forward to find the next 
        period or comma to cleanly finish the sentence.
        Includes a safety cap to prevent token overflow.
        
        """
        if not text:
            return ""
        if len(text) <= min_chars:
            return text
            
        # We reached the limit. Let's look at the remaining text to finish the thought.
        remainder = text[min_chars:]
        
        # Find the first period or comma AFTER our limit
        period_idx = remainder.find('.')
        comma_idx = remainder.find(',')
        
        # Filter out the -1s (not found) to find the closest valid punctuation
        valid_breaks = [idx for idx in (period_idx, comma_idx) if idx != -1]
        
        if valid_breaks:
            cut_offset = min(valid_breaks)
            
            # Safety Cap: If the next period is hundreds of characters away, 
            # we must force a cut so we don't blow up the Bi-Encoder's token limit.
            if cut_offset <= 100: 
                # +1 to include the punctuation mark itself
                return text[:min_chars + cut_offset + 1]
                
        # Fallback: If no punctuation is found soon, just finish the current word
        space_idx = remainder.find(' ')
        if space_idx != -1 and space_idx < 50:
            return text[:min_chars + space_idx] + "..."
            
        # Absolute fallback if it's a giant block of unbroken text without spaces
        return text[:min_chars] + "..."

    def _extract_high_signal_sentences(self, text: str, is_summary: bool = False) -> dict:
        if not text:
            return {"text": "", "impact": [], "routine": [], "trash": []}

        # Safe sentence splitting (Lookahead for capital letter)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]

        impact_bucket = []
        routine_bucket = []
        trash_bucket = []

        # UPGRADE 1: Target subjective phrases instead of single words
        fluff_pattern = re.compile(
            r'\b(aspiring to|eager to learn|passionate about|looking for an opportunity|'
            r'enthusiastic about|seeking to|highly motivated|team player)\b', 
            re.IGNORECASE
        )

        # The Impact Filter (Remains the same)
        impact_pattern = re.compile(r'[\%\$]|\b\d+x\b|\b\d+[KkMmBbTtGg]\b|\b\d{2,}\b')

        for sentence in sentences:
            # UPGRADE 2: Impact Immunity. Check for metrics FIRST.
            # If they reduced costs by 50%, we keep it even if they say they are "passionate about" it.
            if impact_pattern.search(sentence):
                impact_bucket.append(sentence)
                continue

            # Step 2: Fluff Shredding (Only applies to non-metric sentences in the summary)
            if is_summary and fluff_pattern.search(sentence):
                trash_bucket.append(sentence)
                continue

            # Step 3: Default to Routine
            routine_bucket.append(sentence)

        processed_text = " ".join(impact_bucket + routine_bucket)

        return {
            "text": processed_text,
            "impact": impact_bucket,
            "routine": routine_bucket,
            "trash": trash_bucket
        }

    def build_dense_string(self, candidate: dict) -> str:
        """
        Builds a highly compressed, temporally-weighted text string.
        Optimized for the token limit of the BGE Bi-Encoder.
        """
        p = candidate.get('profile', {})
        parts = []

        parts.append(f"HEADLINE: {p.get('headline', '')}")
        
        # Fluff-Shred the Summary
        clean_summary = self._extract_high_signal_sentences(p.get('summary', ''), is_summary=True)["text"]
        if clean_summary:
            parts.append(f"SUMMARY: {clean_summary}")

        career = candidate.get('career_history', [])
        if career:
            recent_job = career[0]
            parts.append(f"CURRENT ROLE: {recent_job.get('title', '')} at {recent_job.get('company', '')} ({recent_job.get('industry', '')})")
            
            # Positionally Prime the job description, THEN truncate
            primed_desc = self._extract_high_signal_sentences(recent_job.get('description', ''), is_summary=False)["text"]
            truncated_desc = self._smart_truncate(primed_desc, 300)
            parts.append(f"CURRENT RESPONSIBILITIES: {truncated_desc}") 

        raw_skills = candidate.get('skills', [])
        if raw_skills:
            sorted_skills = sorted(
                raw_skills, 
                key=lambda x: x.get('duration_months', 0) if x.get('duration_months') is not None else 0, 
                reverse=True
            )
            verified_skills = [
                s['name'] for s in sorted_skills
                if s.get('proficiency') in ('advanced', 'expert') or (s.get('duration_months', 0) if s.get('duration_months') is not None else 0) >= 6
            ]
            if verified_skills:
                parts.append("VERIFIED CORE SKILLS: " + ", ".join(verified_skills))

        if len(career) > 1:
            parts.append("PAST EXPERIENCE:")
            for job in career[1:3]:
                # Positionally Prime past jobs, THEN truncate
                primed_past = self._extract_high_signal_sentences(job.get('description', ''), is_summary=False)["text"]
                truncated_past = self._smart_truncate(primed_past, 150)
                parts.append(f"Previously {job.get('title', '')} at {job.get('company', '')}. {truncated_past}")

        edu_strs = [f"{e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution', '')} ({e.get('tier', 'Unknown')})" 
                    for e in candidate.get('education', [])]
        if edu_strs:
            parts.append("EDUCATION: " + " | ".join(edu_strs))

        return " ".join(parts)

    def build_rich_string(self, candidate: dict) -> str:
        """
        Builds a comprehensive, highly detailed text string.
        Optimized for the deep semantic matching of a Cross-Encoder.
        """
        p = candidate.get('profile', {})
        parts = []

        # 1. Identity & Full Summary
        parts.append(f"CANDIDATE: {p.get('headline', '')}")
        if p.get('summary'):
            parts.append(f"SUMMARY: {p.get('summary')}")

        # 2. Comprehensive Skills (Sorted, but not filtered)
        raw_skills = candidate.get('skills', [])
        if raw_skills:
            # Sort by duration to keep the best stuff first
            sorted_skills = sorted(
                raw_skills, 
                key=lambda x: x.get('duration_months', 0) if x.get('duration_months') is not None else 0, 
                reverse=True
            )
            # Include everything, mapping out their entire technical footprint
            skill_strings = [
                f"{s['name']} ({s.get('proficiency', 'familiar')}, {s.get('duration_months', 0)}mo)" 
                for s in sorted_skills
            ]
            parts.append("ALL SKILLS: " + ", ".join(skill_strings))

        # 3. Full Career History (With Positional Priming, No Truncation)
        career = candidate.get('career_history', [])
        if career:
            parts.append("CAREER HISTORY:")
            for idx, job in enumerate(career):
                # We cap it at the last 4 jobs to avoid exceeding standard Cross-Encoder windows
                if idx >= 4:
                    break
                    
                job_header = f"- {job.get('title', '')} at {job.get('company', '')} ({job.get('industry', '')})"
                
                # Apply the regex sorter to pull metrics up, but we don't truncate the output
                primed_desc = self._extract_high_signal_sentences(job.get('description', ''), is_summary=False)["text"]
                
                if primed_desc:
                    parts.append(f"{job_header}: {primed_desc}")
                else:
                    parts.append(job_header)

        # 4. Education & Extracurriculars
        edu_strs = [f"{e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution', '')} (Tier: {e.get('tier', 'Unknown')})" 
                    for e in candidate.get('education', [])]
        if edu_strs:
            parts.append("EDUCATION: " + " | ".join(edu_strs))

        return "\n".join(parts)

    # Methods Below this are still under work

    def extract_behavioral_signals(self, candidate: dict) -> dict:
        """
        Extracts strict Redrob telemetry signals and calculates derived metrics 
        (like days_inactive) for the Availability Multiplier.
        """
        telemetry = candidate.get('redrob_signals', {})

        # Safely calculate days inactive
        last_active_str = telemetry.get('last_active_date')
        days_inactive = 0  # Default to 0 if missing (assume active)
        
        if last_active_str:
            try:
                # .split('T')[0] handles ISO timestamps if they include time (e.g., 2025-10-12T14:30:00)
                last_active = datetime.datetime.strptime(last_active_str.split('T')[0], '%Y-%m-%d').date()
                days_inactive = max(0, (self.ref_date - last_active).days)
            except ValueError:
                # If the date is totally unparseable, default to a high penalty to be safe
                days_inactive = 180 

        return {
            "telemetry": {
                # The "Ghosting" / Reliability Metrics
                "days_inactive": days_inactive,
                "response_rate": telemetry.get('recruiter_response_rate', 0.0),
                "response_speed_hrs": telemetry.get('avg_response_time_hours', 72),
                "interview_completion": telemetry.get('interview_completion_rate', 0.0),
                "offer_acceptance": telemetry.get('offer_acceptance_rate', -1.0), 
                
                # The "Intent" / Activity Metrics
                "is_active": telemetry.get('open_to_work_flag', False),
                "saved_30d": telemetry.get('saved_by_recruiters_30d', 0),
                "apps_30d": telemetry.get('applications_submitted_30d', 0),
                
                # Platform Quality Metrics
                "github_score": telemetry.get('github_activity_score', -1),
                "profile_completeness": telemetry.get('profile_completeness_score', 0)
            },
            "raw_telemetry": telemetry
        }
        
        # Match relevancy method was removed due to being of the nature hardcoded 