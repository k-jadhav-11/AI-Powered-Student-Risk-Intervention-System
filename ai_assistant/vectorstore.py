
import hashlib
from functools import lru_cache

import numpy as np
from langsmith import traceable

STARTER_REMEDIAL_CONTENT = [
    {"id": "rem_001", "factor_tags": ["failures", "g1", "g2"],
     "text": "Struggling with algebra or core subject basics: review foundational concepts and practice "
             "with step-by-step worked examples before moving to word problems or advanced material."},
    {"id": "rem_002", "factor_tags": ["engagement_trend", "num_weeks_active", "days_active_mean"],
     "text": "Low or declining engagement in an online course: set a fixed weekly check-in time, "
             "even just 15 minutes, since consistency matters more than session length."},
    {"id": "rem_003", "factor_tags": ["pct_assessments_submitted"],
     "text": "Missed multiple assignment deadlines: talk to the instructor about a "
             "catch-up plan before the gap becomes too large to close in one push."},
    {"id": "rem_004", "factor_tags": ["failures"],
     "text": "Repeated course failures: consider a study group or peer tutor for the "
             "specific topics that keep coming up as weak points, rather than restarting cold."},
    {"id": "rem_005", "factor_tags": ["absences"],
     "text": "High or rising absences: identify the specific barrier (schedule "
             "conflict, health, motivation) before assuming more of the same intervention will help."},
    {"id": "rem_006", "factor_tags": ["g1", "g2"],
     "text": "Low first- or second-period grades (g1/g2): schedule a short check-in with the subject "
             "teacher to pinpoint exactly which topic the grade drop traces back to, rather than "
             "reviewing the whole syllabus broadly."},
    {"id": "rem_007", "factor_tags": ["studytime"],
     "text": "Low reported weekly study time: a fixed, short daily block (even 20-30 minutes) tends to "
             "stick better than an irregular longer session once or twice a week."},
    {"id": "rem_008", "factor_tags": ["goout", "dalc", "walc"],
     "text": "High social-outing or leisure-alcohol-use scores alongside falling grades: this is worth "
             "a supportive, non-judgmental conversation about balance rather than a purely academic "
             "intervention -- the risk factor here is about time and attention, not ability."},
    {"id": "rem_009", "factor_tags": ["num_weeks_active", "total_clicks_sum", "days_active_sum"],
     "text": "Very few active weeks on the course platform: a short welcome-back message plus a specific, "
             "small first task (not a full module) lowers the barrier to re-engaging more than a "
             "general reminder does."},
    {"id": "rem_010", "factor_tags": ["engagement_trend"],
     "text": "Engagement trending downward over the course: early terms usually predict this better than "
             "later ones, so a check-in as soon as a downward trend appears is more effective than "
             "waiting for a bigger drop."},
    {"id": "rem_011", "factor_tags": ["schoolsup_yes", "famsup_yes"],
     "text": "No school or family educational support currently in place: connecting the student with "
             "existing school support resources (tutoring, counseling) is often the highest-leverage "
             "single step available."},
    {"id": "rem_012", "factor_tags": ["absences", "num_weeks_active"],
     "text": "Persona note -- students matching a 'quietly disengaged' or 'repeat-attempt struggling' "
             "profile (see the Clustering Explorer / Performance Personas page) tend to respond better "
             "to a proactive, low-pressure check-in than to a formal warning, since the pattern is "
             "usually disengagement rather than active failure."},

    {"id": "rem_013", "factor_tags": ["traveltime"],
         "text": "Long home-to-school travel time: check whether a schedule adjustment, remote/async "
                 "option for some sessions, or transit assistance is available before assuming the "
                 "commute itself can't be shortened."},
    {"id": "rem_014", "factor_tags": ["freetime"],
         "text": "High reported free time alongside falling grades: this often reflects unstructured "
                 "time rather than laziness -- a simple weekly planning template can help convert some "
                 "of that free time into a fixed study block."},
    {"id": "rem_015", "factor_tags": ["famrel"],
         "text": "Lower reported quality of family relationships: loop in a school counselor before an "
                 "academic-only intervention, since this factor points at a home-context stressor an "
                 "academic plan alone won't address."},
    {"id": "rem_016", "factor_tags": ["medu", "fedu"],
         "text": "Lower parental education levels: this student may have less at-home academic support "
                 "available by default, so school-side scaffolding (structured homework help, study "
                 "hall) can matter more than for peers with more at-home support."},
    {"id": "rem_017", "factor_tags": ["health"],
         "text": "Lower self-reported health status: rule out a health-related barrier to attendance or "
                 "concentration before assuming the academic pattern is motivational."},
    {"id": "rem_018", "factor_tags": ["age"],
         "text": "Age notably older or younger than the typical cohort for this grade: check whether "
                 "this reflects a prior repeated year or an accelerated placement, since either changes "
                 "what 'normal progress' looks like for this specific student."},
    {"id": "rem_019", "factor_tags": ["internet_yes"],
         "text": "No reliable home internet access: for any online-component coursework, confirm the "
                 "school can offer on-site access (library, lab hours) rather than assuming take-home "
                 "digital work is equally accessible to this student."},
    {"id": "rem_020", "factor_tags": ["higher_yes"],
         "text": "Not currently aspiring to pursue higher education: a low-pressure conversation about "
                 "post-graduation options (not necessarily university) can re-engage a student who has "
                 "quietly stopped seeing the point of current coursework."},
    {"id": "rem_021", "factor_tags": ["romantic_yes"],
         "text": "Currently in a romantic relationship alongside falling grades: worth a supportive, "
                 "non-judgmental time-management conversation -- similar in spirit to the "
                 "social-outing/leisure factor, this is about attention and time, not ability."},
    {"id": "rem_022", "factor_tags": ["famsize_LE3", "address_U", "school_MS", "sex_M"],
         "text": "Background/demographic context (family size, urban vs. rural address, school, sex): "
                 "these describe the student's situation, not a lever to pull directly -- use them only "
                 "to interpret other factors (e.g. a long traveltime reading differently for a rural "
                 "address), never as a target for intervention on their own."},
    {"id": "rem_023", "factor_tags": ["subject_Portuguese"],
         "text": "Course/subject context (Portuguese vs. Math track): confirm which subject's grading "
                 "pattern the risk factors actually reflect before generalizing a recommendation to the "
                 "student's other coursework."},
    {"id": "rem_024", "factor_tags": ["total_clicks_mean", "total_clicks_max"],
         "text": "Engagement that spikes around deadlines rather than staying steady (high max, low "
                 "mean weekly clicks): a light structured reminder a few days before each deadline tends "
                 "to smooth this pattern out better than a single end-of-module nudge."},
    {"id": "rem_025", "factor_tags": ["num_of_prev_attempts"],
         "text": "Multiple previous attempts at this same course: ask what specifically stalled the "
                 "prior attempt(s) before repeating the same support plan -- a repeat attempt with the "
                 "same intervention that already failed once is unlikely to succeed a second time."},
    {"id": "rem_026", "factor_tags": ["studied_credits"],
         "text": "A high number of credits currently being studied at once: a course-load conversation "
                 "(is this student overcommitted this term) is worth having before assuming any single "
                 "course's content is the problem."},
    {"id": "rem_027", "factor_tags": ["highest_education_ordinal"],
         "text": "Lower highest-prior-education level entering this course: this student may benefit "
                 "from an early foundations/bridging module rather than the standard pacing assumed for "
                 "students entering with more prior formal education."},
    {"id": "rem_028", "factor_tags": ["age_band_ordinal"],
         "text": "Age band notably different from the course's typical cohort: check whether this "
                 "student is balancing work or family responsibilities the standard course pacing "
                 "doesn't account for, rather than assuming disengagement."},
    {"id": "rem_029", "factor_tags": ["imd_band_ordinal", "imd_band_missing"],
         "text": "Home-area deprivation index is high, or missing entirely from the record: a high "
                 "value is worth a referral to existing student-support/financial-aid resources; a "
                 "missing value should first be treated as a data gap to flag, not read as a risk "
                 "signal on its own."},
    {"id": "rem_030", "factor_tags": ["code_module_BBB", "code_module_CCC", "code_module_DDD",
                                           "code_module_EEE", "code_module_FFF", "code_module_GGG"],
         "text": "Which course module this record belongs to: compare this student's pattern against "
                 "that specific module's own typical pass/fail rate (see the Risk Dashboard's per-module "
                 "breakdown) before assuming the same intervention that works in one module transfers "
                 "directly to another."},
    {"id": "rem_031", "factor_tags": ["region_East Midlands Region", "region_Ireland", "region_London Region",
                                           "region_North Region", "region_North Western Region", "region_Scotland",
                                           "region_South East Region", "region_South Region", "region_South West Region",
                                           "region_Wales", "region_West Midlands Region", "region_Yorkshire Region"],
         "text": "Home region: mainly useful for spotting a genuine regional pattern across many "
                 "students (e.g. a region-wide access or connectivity issue) rather than as an "
                 "individual-student action item on its own."},
    {"id": "rem_032", "factor_tags": ["gender_M"],
         "text": "Gender: use only to check whether a support program is reaching students evenly "
                 "across genders in the cohort view -- not a basis for an individual student's "
                 "intervention plan."},
    {"id": "rem_033", "factor_tags": ["disability_Y"],
         "text": "A declared disability on record: confirm the student is actually connected to the "
                 "accommodations they're entitled to (extended time, accessible materials) before any "
                 "other intervention -- an academic nudge is the wrong first step if accommodations "
                 "aren't already in place."},
    {"id": "rem_034", "factor_tags": ["failures", "g1", "g2", "absences"],
         "text": "Compounding academic risk (repeated failures plus low grades plus rising absences "
                 "together, not just one factor alone): prioritize this student for a human mentor "
                 "conversation over a self-serve resource -- multiple reinforcing factors are exactly "
                 "the case a lightweight nudge is least likely to be enough for."},
    {"id": "rem_035", "factor_tags": ["absences", "pct_assessments_submitted"],
         "text": "Rising absences alongside missed assessment submissions: these two usually share one "
                 "root cause -- confirm the student can actually access the material and submission "
                 "system before treating them as two separate problems needing two separate fixes."},
    {"id": "rem_036", "factor_tags": ["dalc", "walc", "health"],
         "text": "Elevated alcohol-use scores alongside lower self-reported health: treat this as a "
                 "wellbeing signal to route to a counselor first, not an academic performance issue to "
                 "solve with study tips."},
    {"id": "rem_037", "factor_tags": ["engagement_trend", "total_clicks_max"],
         "text": "Inconsistent engagement -- occasional high-activity spikes but an overall declining "
                 "trend: this pattern often means the student re-engages under pressure but doesn't "
                 "sustain it; a standing weekly check-in tends to help more than deadline-only reminders."},
    {"id": "rem_038", "factor_tags": ["num_of_prev_attempts", "studied_credits"],
         "text": "A repeat attempt combined with a high current credit load: consider whether this "
                 "student would be better served retaking this course alone next term rather than "
                 "alongside a full course load again."},
]

_EMBED_DIM = 64


def _hash_embedding(text: str, dim: int = _EMBED_DIM) -> list[float]:
    """Creates a deterministic hash-based embedding for offline fallback.
    Returns a normalized vector with the specified dimensionality."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


@lru_cache(maxsize=1)
@traceable(name="load_sentence_transformer_model", run_type="tool")
def _load_sentence_transformer():
    """Loads the SentenceTransformer model once and reuses it across calls.
    Raises an exception if the model cannot be loaded."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding_function():
    """Returns an embedding function and the backend used to generate embeddings.
    Uses SentenceTransformer when available, otherwise falls back to hash embeddings."""
    try:
        model = _load_sentence_transformer()

        def embed_fn(texts: list[str]) -> list[list[float]]:
            return model.encode(texts).tolist()

        return embed_fn, "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        def embed_fn(texts: list[str]) -> list[list[float]]:
            return [_hash_embedding(t) for t in texts]

        return embed_fn, "hash-fallback (offline placeholder, not semantic)"


def build_chroma_collection(collection_name: str = "remedial_content_starter"):
    """Builds an in-memory ChromaDB collection from the starter remedial content.
    Returns the collection and the embedding backend used."""
    import chromadb

    embed_fn, backend = get_embedding_function()
    client = chromadb.Client()  # ephemeral, in-memory client
   
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    embeddings = embed_fn([item["text"] for item in STARTER_REMEDIAL_CONTENT])
    collection.add(
        ids=[item["id"] for item in STARTER_REMEDIAL_CONTENT],
        documents=[item["text"] for item in STARTER_REMEDIAL_CONTENT],
        embeddings=embeddings,
    )
    return collection, backend


def build_faiss_index():
    """Builds a FAISS IndexFlatL2 index from the starter remedial content.
    Returns the index and the embedding backend used."""
    import faiss

    embed_fn, backend = get_embedding_function()
    embeddings = np.array(embed_fn([item["text"] for item in STARTER_REMEDIAL_CONTENT]), dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, backend


def verify_pipeline() -> dict:
    """Runs the ChromaDB and FAISS embedding pipelines as an end-to-end check.
    Returns backend names and document counts for both vector stores."""
    collection, chroma_backend = build_chroma_collection()
    index, faiss_backend = build_faiss_index()
    return {
        "chroma_backend": chroma_backend,
        "chroma_count": collection.count(),
        "faiss_backend": faiss_backend,
        "faiss_count": index.ntotal,
        "starter_content_count": len(STARTER_REMEDIAL_CONTENT),
    }