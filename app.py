import os
import sqlite3
import pandas as pd
import hashlib
import secrets
import streamlit as st
from datetime import datetime, timedelta
import pathlib

# Optional Gemini import
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ================= Configuration =================
# WARNING: Replace this placeholder with a real key in a secure manner for production
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCn9HTzFrUX0quRSFQ9WUg4EcQxvqSImZg")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DB_PATH = pathlib.Path(__file__).with_name("ai_tutor.db").as_posix()
PASSWORD_SALT = os.getenv("AI_TUTOR_SALT", "change_this_salt")

st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
body {background-color: #f8f9fa;}
.main {background-color: #ffffff; border-radius: 10px; padding: 30px; box-shadow: 0px 2px 10px rgba(0,0,0,0.1);}
h1,h2,h3 {color: #1a237e;}
.stButton>button {background-color: #3949ab; color:white; border-radius:8px; height:3em; width:100%;}
.stButton>button:hover {background-color:#283593; color:white;}
</style>
""", unsafe_allow_html=True)

# ================= Database Setup (cached, single connection) =================
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

conn = get_conn()
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    fullname TEXT,
    role TEXT,
    password_hash TEXT,
    salt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved INTEGER DEFAULT 0
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS question_bank (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT,
    difficulty INTEGER,
    prompt TEXT,
    answer TEXT
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    subject TEXT,
    question TEXT,
    model_answer TEXT,
    user_answer TEXT,
    score REAL,
    feedback TEXT,
    ai_feedback TEXT,
    rating INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# ================= Utility Functions =================
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    salted = (password + PASSWORD_SALT + salt).encode('utf-8')
    h = hashlib.sha256(salted).hexdigest()
    return h, salt

def verify_password(password, stored_hash, salt):
    h, _ = hash_password(password, salt)
    return h == stored_hash

def create_user(username, fullname, role, password, approved=0):
    h, salt = hash_password(password)
    try:
        cur.execute(
            "INSERT INTO users (username, fullname, role, password_hash, salt, approved) VALUES (?,?,?,?,?,?)",
            (username, fullname, role, h, salt, approved)
        )
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "⚠️ Username already exists."

def authenticate(username, password):
    cur.execute("SELECT id, username, fullname, role, password_hash, salt, approved FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        return False, "Not registered. Please register first."
    uid, uname, fullname, role, p_hash, salt, approved = row
    if not verify_password(password, p_hash, salt):
        return False, "Incorrect password."
    if role == 'student' and not approved:
        return False, "Awaiting teacher approval."
    return True, {"id": uid, "username": uname, "fullname": fullname, "role": role}

def call_gemini(prompt):
    if not genai or GEMINI_API_KEY == "AIzaSyCn9HTzFrUX0quRSFQ9WUg4EcQxvqSImZg":
        return "[Gemini not configured or API Key is missing. Please set a valid GEMINI_API_KEY.]"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, 'text') else str(resp)
    except Exception as e:
        return f"[Gemini error: {e}]"

def get_quiz_questions(subject, difficulty, limit=5):
    cur.execute(
        "SELECT id, prompt, answer FROM question_bank WHERE subject=? AND difficulty=? ORDER BY RANDOM() LIMIT ?",
        (subject, difficulty, limit)
    )
    return cur.fetchall()

def generate_weekly_test(subject, difficulty_level, num_questions):
    cur.execute(
        "SELECT id, prompt, answer FROM question_bank WHERE subject=? AND difficulty=? ORDER BY RANDOM() LIMIT ?",
        (subject, difficulty_level, num_questions)
    )
    return cur.fetchall()

# ================= Streamlit App Logic =================
def register_ui():
    st.subheader("Register")
    username = st.text_input("Username")
    fullname = st.text_input("Full Name")
    role = st.selectbox("Role", ["student", "admin"])
    password = st.text_input("Password", type="password")
    if st.button("Register"):
        if username and password:
            approved = 1 if role == "admin" else 0
            success, msg = create_user(username, fullname, role, password, approved)
            st.info(msg)
        else:
            st.warning("Please fill all fields!")

def login_ui():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        success, result = authenticate(username, password)
        if success:
            st.session_state["user"] = result
            st.session_state["page"] = "dashboard"
            if 'student_test_taken' not in st.session_state:
                st.session_state['student_test_taken'] = {}
            if 'weekly_test' not in st.session_state:
                st.session_state['weekly_test'] = None
        else:
            st.warning(result)

# ================= Admin Dashboard =================
def admin_dashboard(user):
    st.title(f"Welcome, {user['fullname']} (Admin)")

    tab1, tab2, tab3, tab4 = st.tabs(["Approve Students","Manage Questions","Bulk Upload","View Progress"])

    # ---- TAB 1: Approvals ----
    with tab1:
        st.subheader("✅ Student Approval")
        cur.execute("SELECT id, username, fullname FROM users WHERE role='student' AND approved=0")
        rows = cur.fetchall()
        if rows:
            for r in rows:
                st.write(f"**{r[2]}** (@{r[1]})")
                if st.button(f"Approve {r[1]}", key=f"approve_{r[0]}"):
                    cur.execute("UPDATE users SET approved=1 WHERE id=?", (r[0],))
                    conn.commit()
                    st.success(f"{r[1]} approved!")
                    st.rerun()
        else:
            st.info("No students awaiting approval.")

    # ---- TAB 2: Manage Questions (enhanced editor with auto-clear) ----
    with tab2:
        st.write("### ➕ Add / Edit Questions")

        SUBJECTS = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"]
        fc1, fc2, fc3 = st.columns([2,1,1])
        with fc1:
            filter_subject = st.selectbox("Filter by Subject", SUBJECTS, key="adm_filter_subject")
        with fc2:
            filter_difficulty = st.selectbox("Filter by Difficulty", ["All", 1, 2, 3], key="adm_filter_diff")
        with fc3:
            st.write("")
            if st.button("↻ Refresh List"):
                st.rerun()

        # Initialize editor state + previous filters
        if "editor_state" not in st.session_state:
            st.session_state.editor_state = {"id": None, "subject": SUBJECTS[0], "difficulty": 1, "prompt": "", "answer": ""}
        if "prev_filter_subject" not in st.session_state:
            st.session_state.prev_filter_subject = filter_subject
        if "prev_filter_diff" not in st.session_state:
            st.session_state.prev_filter_diff = filter_difficulty

        # Auto-clear editor when filters change (new mode)
        filter_changed = (
            st.session_state.prev_filter_subject != filter_subject
            or st.session_state.prev_filter_diff != filter_difficulty
        )
        if filter_changed:
            st.session_state.editor_state.update({
                "id": None,
                "subject": filter_subject,
                "difficulty": 1 if filter_difficulty == "All" else int(filter_difficulty),
                "prompt": "",
                "answer": ""
            })
            st.session_state.prev_filter_subject = filter_subject
            st.session_state.prev_filter_diff = filter_difficulty

        # Load filtered questions
        if filter_difficulty == "All":
            df_list = pd.read_sql_query(
                "SELECT id, subject, difficulty, prompt, answer FROM question_bank WHERE subject=? ORDER BY id DESC",
                conn, params=(filter_subject,)
            )
        else:
            df_list = pd.read_sql_query(
                "SELECT id, subject, difficulty, prompt, answer FROM question_bank WHERE subject=? AND difficulty=? ORDER BY id DESC",
                conn, params=(filter_subject, int(filter_difficulty))
            )

        st.markdown("#### 📘 Existing Questions (filtered)")
        if df_list.empty:
            st.info("No questions found for this filter.")
        else:
            st.dataframe(df_list[["id","subject","difficulty","prompt","answer"]],
                         use_container_width=True, height=240)

        # Helper to load a row into editor
        def load_into_editor(row):
            st.session_state.editor_state = {
                "id": int(row["id"]),
                "subject": row["subject"],
                "difficulty": int(row["difficulty"]),
                "prompt": row["prompt"] or "",
                "answer": row["answer"] or "",
            }

        # Pick and load existing row
        if not df_list.empty:
            with st.expander("🔍 Load an existing question into the editor"):
                pick_id = st.selectbox("Select Question ID to edit",
                                       df_list["id"].astype(int).tolist(),
                                       key="adm_pick_qid")
                if st.button("Load Selected Question"):
                    row = df_list[df_list["id"] == pick_id].iloc[0]
                    load_into_editor(row)
                    st.success(f"Loaded question #{pick_id} into editor")

        st.markdown("---")

        # Editor Form
        st.markdown("#### ✏️ Question Editor")
        with st.form(key="adm_editor_form"):
            ec1, ec2 = st.columns([2,1])
            with ec1:
                subj_val = st.selectbox(
                    "Subject",
                    SUBJECTS,
                    index=SUBJECTS.index(st.session_state.editor_state["subject"])
                )
            with ec2:
                diff_val = st.selectbox(
                    "Difficulty",
                    [1,2,3],
                    index=[1,2,3].index(st.session_state.editor_state["difficulty"])
                )

            prompt_val = st.text_area(
                "Question Prompt",
                value=st.session_state.editor_state["prompt"],
                height=120,
                key="adm_editor_prompt"
            )
            answer_val = st.text_area(
                "Answer (optional)",
                value=st.session_state.editor_state["answer"],
                height=120,
                key="adm_editor_answer"
            )

            c_new, c_add, c_upd = st.columns([1,1,1])
            do_new = c_new.form_submit_button("🆕 New Question")
            do_add = c_add.form_submit_button("➕ Add New")
            do_upd = c_upd.form_submit_button("💾 Update Current")

        # Sync editor state with current form values
        st.session_state.editor_state.update({
            "subject": subj_val, "difficulty": diff_val,
            "prompt": prompt_val, "answer": answer_val
        })

        # Actions
        if do_new:
            st.session_state.editor_state.update({"id": None, "prompt": "", "answer": ""})
            st.success("Editor cleared for a new question.")

        if do_add:
            if not prompt_val.strip():
                st.warning("Please enter a question prompt before adding.")
            else:
                cur.execute(
                    "INSERT INTO question_bank (subject, difficulty, prompt, answer) VALUES (?,?,?,?)",
                    (subj_val, diff_val, prompt_val, answer_val)
                )
                conn.commit()
                st.success("✅ Question added!")
                st.session_state.editor_state.update({"id": None, "prompt": "", "answer": ""})
                st.rerun()

        if do_upd:
            qid = st.session_state.editor_state["id"]
            if qid is None:
                st.warning("No question loaded. Use the loader above to pick a question first.")
            else:
                cur.execute(
                    "UPDATE question_bank SET subject=?, difficulty=?, prompt=?, answer=? WHERE id=?",
                    (subj_val, diff_val, prompt_val, answer_val, int(qid))
                )
                conn.commit()
                st.success(f"✅ Question #{qid} updated!")
                st.rerun()

        # Delete action (separate button)
        if st.session_state.editor_state["id"] is not None:
            dc1, _ = st.columns([1,3])
            with dc1:
                if st.button("🗑️ Delete This Question"):
                    qid = int(st.session_state.editor_state["id"])
                    cur.execute("DELETE FROM question_bank WHERE id=?", (qid,))
                    conn.commit()
                    st.success(f"🗑️ Deleted question #{qid}")
                    st.session_state.editor_state.update({"id": None, "prompt": "", "answer": ""})
                    st.rerun()

        st.markdown("---")
        st.subheader("🗓️ Generate Weekly Test")
        st.info("Set the parameters and generate a new weekly test for all students.")
        test_subject = st.selectbox("Test Subject", SUBJECTS, key="test_subj")
        test_difficulty = st.selectbox("Average Difficulty", [1, 2, 3], key="test_diff")
        test_num_questions = st.slider("Number of Test Questions", 5, 20, 10)

        if st.button("SET THIS AS WEEKLY TEST", type="primary"):
            questions = generate_weekly_test(test_subject, test_difficulty, test_num_questions)
            if questions:
                st.session_state['weekly_test'] = {
                    'date': datetime.now().date().isoformat(),
                    'subject': test_subject,
                    'difficulty': test_difficulty,
                    'questions': questions
                }
                st.session_state['student_test_taken'] = {}
                st.session_state['test_final_score'] = None
                st.success(f"✅ **Weekly Test Generated!** ({len(questions)} questions from {test_subject}, Difficulty {test_difficulty})")
            else:
                st.error("Could not generate test. Check if enough questions exist in the **Question Bank**.")

        st.markdown("---")
        st.write("### 📘 All Questions (no filter)")
        df_all = pd.read_sql_query("SELECT * FROM question_bank ORDER BY id DESC", conn)
        st.dataframe(df_all, use_container_width=True)

    # ---- TAB 3: Bulk Upload ----
    with tab3:
        st.subheader("📥 Bulk Upload Questions (CSV)")
        csv_file = st.file_uploader("Upload CSV with columns: subject,difficulty,prompt,answer", type=["csv"])
        if csv_file is not None:
            try:
                df_bulk = pd.read_csv(csv_file)
                df_bulk.columns = df_bulk.columns.str.lower()
                expected = {"subject", "difficulty", "prompt", "answer"}
                if not expected.issubset(set(df_bulk.columns)):
                    st.error("CSV must contain columns: subject,difficulty,prompt,answer")
                else:
                    inserted = 0
                    for _, r in df_bulk.iterrows():
                        subj = r["subject"]
                        diff = int(r["difficulty"] or 1)
                        prompt_txt = r["prompt"]
                        ans_txt = r.get("answer", "")
                        cur.execute(
                            "INSERT INTO question_bank (subject, difficulty, prompt, answer) VALUES (?,?,?,?)",
                            (subj, diff, prompt_txt, ans_txt)
                        )
                        inserted += 1
                    conn.commit()
                    st.success(f"Inserted {inserted} questions.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # ---- TAB 4: View Progress ----
    with tab4:
        st.write("### Student Performance Overview")
        df = pd.read_sql_query("""
            SELECT s.id, s.user_id, u.fullname, u.username, s.subject, s.score, s.feedback, s.rating, s.created_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            ORDER BY s.created_at DESC
        """, conn)

        if df.empty:
            st.info("No student sessions recorded yet.")
        else:
            df["Score (%)"] = (df["score"].fillna(0) * 100).round(1)
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Attempts", len(df))
            c2.metric("Avg Score", f"{df['Score (%)'].mean():.1f}%")
            c3.metric("Unique Students", df["username"].nunique())

            st.dataframe(
                df[["created_at", "fullname", "username", "subject", "Score (%)", "feedback", "rating"]]
                  .sort_values("created_at", ascending=False),
                use_container_width=True
            )

# ================= Student Dashboard =================
def student_dashboard(user):
    st.title(f"👩‍🎓 Welcome, {user['fullname']}")

    cur.execute("SELECT created_at, score FROM sessions WHERE user_id=? ORDER BY created_at DESC", (user["id"],))
    rows = cur.fetchall()

    dates = []
    for created_at, _ in rows:
        try:
            dt = pd.to_datetime(created_at, errors="coerce")
            if pd.notnull(dt):
                dates.append(dt.date())
        except Exception:
            pass

    dates = sorted(set(dates), reverse=True)
    streak = 0
    today = datetime.now().date()
    for i in range(365):
        if (today - timedelta(days=i)) in dates:
            streak += 1
        else:
            break

    st.metric("🔥 Current Streak", f"{streak} days")

    badges = []
    if streak >= 3:
        badges.append("🔥 3-Day Streak")
    if streak >= 7:
        badges.append("💎 7-Day Streak")
    if len(rows) >= 20:
        badges.append("🏅 20+ Sessions")

    scores = [s for _, s in rows if s is not None]
    avg_score = (sum(scores) / len(scores)) if scores else 0
    if avg_score > 0.8:
        badges.append("🌟 High Achiever")

    if badges:
        st.info("🏆 Your Badges: " + ", ".join(badges))
    else:
        st.info("No badges yet — start practicing!")

    tab1, tab2, tab3, tab4 = st.tabs(["Ask AI Tutor", "Weekly Test", "Quiz Me", "Previous Sessions"])

    # --- Tab 1: Ask AI Tutor (Chat) ---
    with tab1:
        st.subheader("🗣️ Ask the Tutor Anything")
        subject = st.selectbox("Subject", ["Mathematics", "Physics", "Chemistry"], key="ai_tutor_subject")
        question = st.text_area("Enter your question", key="ai_tutor_question")

        if st.button("Ask Tutor"):
            if question.strip():
                for key in ["quiz_questions", "current_q_index", "quiz_score", 'test_attempt']:
                    if key in st.session_state:
                        del st.session_state[key]

                with st.spinner('Generating answer...'):
                    model_ans = call_gemini(
                        f"You are a helpful tutor in {subject}. Provide a step-by-step explanation. Question: {question}"
                    )

                st.session_state["current_question"] = question
                st.session_state["current_subject"] = subject
                st.session_state["model_ans"] = model_ans
                st.success("✅ Tutor Answer Generated!")
                st.rerun()

        if "model_ans" in st.session_state:
            st.write("### 💬 AI Tutor Answer")
            st.success(st.session_state["model_ans"])

            user_ans = st.text_area("✍️ Your Answer / Attempt", key="user_answer_input")
            rating = st.slider("⭐ Rate the question (1-5)", 1, 5, 3, key="rate_slider")
            feedback = st.text_area("💬 Your Feedback (optional)", key="student_feedback_input")

            if st.button("Submit Answer", key="submit_ai_tutor_answer"):
                ma = set(st.session_state["model_ans"].lower().split())
                ua = set(user_ans.lower().split())
                score = len(ma.intersection(ua)) / max(len(ma), 1)

                cur.execute("""
                    INSERT INTO sessions (user_id, subject, question, model_answer, user_answer, score, feedback, rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user['id'],
                    st.session_state["current_subject"],
                    st.session_state["current_question"],
                    st.session_state["model_ans"],
                    user_ans,
                    score,
                    feedback,
                    rating
                ))
                conn.commit()

                st.success(f"✅ Session saved! Score: {score*100:.1f}%")

                for key in ["model_ans", "current_question", "current_subject"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # --- Tab 2: Weekly Test (with empty-answer guard) ---
    with tab2:
        st.subheader("🎯 Weekly Performance Test")
        user_id = user['id']

        if st.session_state['weekly_test'] is None:
            st.info("The teacher has not yet set the weekly test. Check back later!")
        elif st.session_state.get('student_test_taken', {}).get(user_id) == st.session_state['weekly_test']['date']:
            st.warning(f"You have already submitted the test for the week of **{st.session_state['weekly_test']['date']}**.")
            st.success(f"Your last test score: **{st.session_state.get('test_final_score', 'N/A')}**")
        else:
            test_info = st.session_state['weekly_test']
            questions = test_info['questions']
            num_questions = len(questions)

            st.info(f"Test Set: **{test_info['date']}** | **{test_info['subject']}** | {num_questions} Questions")

            st.error("🚨 **Proctoring Mode:** This is a monitored test.")
            st.warning("You must remain visible to the camera. Excessive head turning (more than 3 times) will be logged as a potential violation and may result in test invalidation.")

            if 'test_attempt' not in st.session_state:
                st.session_state['test_attempt'] = {
                    'index': 0,
                    'answers': [''] * num_questions,
                    'scores': [0.0] * num_questions,
                    'in_progress': True
                }

            attempt = st.session_state['test_attempt']
            q_idx = attempt['index']

            if attempt['in_progress'] and q_idx < num_questions:
                q_id, prompt, model_ans = questions[q_idx]

                st.subheader(f"Question {q_idx + 1} of {num_questions}")
                st.markdown(f"**Question:**\n{prompt}")

                user_ans = st.text_area("✍️ Your Test Answer",
                                        value=attempt['answers'][q_idx],
                                        key=f"test_ans_{q_idx}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Previous Question", disabled=(q_idx == 0)):
                        attempt['answers'][q_idx] = user_ans
                        attempt['index'] -= 1
                        st.rerun()
                with col2:
                    if st.button("Next Question / Review"):
                        if not user_ans.strip():
                            st.warning("Please enter an answer before moving ahead.")
                        else:
                            attempt['answers'][q_idx] = user_ans
                            attempt['index'] += 1
                            st.rerun()

            elif attempt['in_progress'] and q_idx == num_questions:
                st.subheader("Review and Submission")
                st.warning("Review your answers before submission. Once submitted, you cannot retake the test.")

                for i in range(num_questions):
                    st.markdown(f"**Q{i+1}:** {questions[i][1][:50]}...")
                    st.text_area(f"Your Answer for Q{i+1}", value=attempt['answers'][i], height=50, disabled=True, key=f"review_ans_{i}")

                if st.button("FINISH AND SUBMIT TEST", type="primary"):
                    total_correct = 0
                    for i in range(num_questions):
                        q_prompt = questions[i][1]
                        model_ans = questions[i][2]
                        user_ans = attempt['answers'][i]

                        ma_words = set(model_ans.lower().split())
                        ua_words = set(user_ans.lower().split())
                        score = len(ma_words.intersection(ua_words)) / max(len(ma_words), 1)

                        attempt['scores'][i] = score
                        if score > 0.5:
                            total_correct += 1

                        cur.execute("""
                            INSERT INTO sessions (user_id, subject, question, model_answer, user_answer, score, feedback, rating)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            user_id,
                            test_info['subject'],
                            q_prompt,
                            model_ans,
                            user_ans,
                            score,
                            "Weekly Test Submission",
                            None
                        ))
                    conn.commit()

                    attempt['in_progress'] = False
                    st.session_state['student_test_taken'][user_id] = test_info['date']
                    st.session_state['test_final_score'] = f"{total_correct}/{num_questions} ({total_correct/num_questions * 100:.1f}%)"
                    del st.session_state['test_attempt']

                    st.success(f"🎉 **Test Submitted!** Your score: {st.session_state['test_final_score']}")
                    st.rerun()

            if 'test_attempt' in st.session_state and st.session_state['test_attempt']['in_progress']:
                st.markdown(f"**Progress:** {st.session_state['test_attempt']['index']} / {num_questions}")

    # --- Tab 3: Quiz Me (Daily Practice, with empty-answer guard) ---
    with tab3:
        st.subheader("📚 Practice Quiz (Daily)")
        if "quiz_questions" not in st.session_state:
            st.session_state["quiz_questions"] = []
            st.session_state["current_q_index"] = 0
            st.session_state["quiz_score"] = 0
            st.session_state["quiz_in_progress"] = False
            st.session_state["quiz_subj"] = "Mathematics"
            st.session_state["quiz_diff"] = 1

        if not st.session_state["quiz_in_progress"]:
            st.subheader("📝 Start a New Practice Quiz")
            st.session_state["quiz_subj"] = st.selectbox("Select Quiz Subject", ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"], key="quiz_subj_select")
            st.session_state["quiz_diff"] = st.selectbox("Select Difficulty", [1, 2, 3], key="quiz_diff_select")
            num_questions = st.slider("Number of Questions", 1, 10, 5, key="quiz_num_q")

            if st.button("Generate Quiz", key="generate_quiz_btn"):
                questions = get_quiz_questions(st.session_state["quiz_subj"], st.session_state["quiz_diff"], num_questions)
                if not questions:
                    st.warning("⚠️ No questions found for this subject and difficulty. Try adding some via the Admin dashboard.")
                else:
                    st.session_state["quiz_questions"] = questions
                    st.session_state["current_q_index"] = 0
                    st.session_state["quiz_score"] = 0
                    st.session_state["quiz_in_progress"] = True
                    st.rerun()

        if st.session_state["quiz_in_progress"] and st.session_state["quiz_questions"]:
            q_index = st.session_state["current_q_index"]
            total_q = len(st.session_state["quiz_questions"])

            if q_index < total_q:
                q_id, prompt, model_ans = st.session_state["quiz_questions"][q_index]

                st.subheader(f"Question {q_index + 1} of {total_q}")
                st.info(f"**Subject:** {st.session_state['quiz_subj']} | **Difficulty:** {st.session_state['quiz_diff']}")
                st.markdown(f"**Question:**\n{prompt}")

                user_ans = st.text_area("✍️ Your Answer", key=f"quiz_ans_{q_index}")

                with st.form(key=f"q_form_{q_index}"):
                    submit_q = st.form_submit_button("Submit & Next Question")

                if submit_q:
                    # --- block empty answers ---
                    if not user_ans.strip():
                        st.warning("Please enter an answer before moving to the next question.")
                    else:
                        # score + save only when there is an answer
                        ma_words = set(model_ans.lower().split())
                        ua_words = set(user_ans.lower().split())
                        score = len(ma_words.intersection(ua_words)) / max(len(ma_words), 1)

                        cur.execute("""
                            INSERT INTO sessions (user_id, subject, question, model_answer, user_answer, score, feedback, rating)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            user['id'],
                            st.session_state["quiz_subj"],
                            prompt,
                            model_ans,
                            user_ans,
                            score,
                            "Practice Quiz Session",
                            None
                        ))
                        conn.commit()

                        if score > 0.5:
                            st.session_state["quiz_score"] += 1
                            st.balloons()

                        st.session_state["current_q_index"] += 1
                        st.rerun()

            else:
                final_score = st.session_state["quiz_score"]
                st.success(f"🎉 Quiz Complete! Your Final Score: **{final_score} / {total_q}**")

                st.session_state["quiz_questions"] = []
                st.session_state["current_q_index"] = 0
                st.session_state["quiz_score"] = 0
                st.session_state["quiz_in_progress"] = False

                if st.button("Start New Quiz", key="new_quiz_after_finish"):
                    st.rerun()

        if st.session_state["quiz_in_progress"]:
            st.markdown(f"**Current Score:** {st.session_state['quiz_score']} / {st.session_state['current_q_index']}")

    # --- Tab 4: View Previous Sessions (History) ---
    with tab4:
        st.write("### 📜 All Session History")
        df = pd.read_sql_query(
            """
            SELECT subject, question, model_answer, user_answer, score, feedback, rating, created_at
            FROM sessions WHERE user_id=? ORDER BY created_at DESC
            """,
            conn,
            params=(user['id'],)
        )
        if df.empty:
            st.info("No previous sessions found.")
        else:
            df["Score (%)"] = (df["score"].fillna(0) * 100).round(1)
            df['question_preview'] = df['question'].fillna("").str.slice(0, 50) + '...'
            st.dataframe(df[["subject", "question_preview", "Score (%)", "feedback", "rating", "created_at"]], use_container_width=True)

# ================= Main App =================
def main():
    if 'student_test_taken' not in st.session_state:
        st.session_state['student_test_taken'] = {}
    if 'weekly_test' not in st.session_state:
        st.session_state['weekly_test'] = None

    # st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🎓 AI Tutor Web App for Tuition Teachers")

    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"]=="home":
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            login_ui()
        with register_tab:
            register_ui()

    elif st.session_state["page"]=="dashboard":
        user = st.session_state.get("user")
        if user["role"]=="admin":
            admin_dashboard(user)
        else:
            student_dashboard(user)

        st.markdown("---")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # # ---- Quick DB Debug Panel ----
    # with st.expander("🔧 Debug: Database snapshot"):
    #     try:
    #         st.write("DB path:", DB_PATH)
    #         u = pd.read_sql_query("SELECT id, username, fullname, role, approved, created_at FROM users ORDER BY id DESC LIMIT 10", conn)
    #         s = pd.read_sql_query("SELECT id, user_id, subject, score, created_at FROM sessions ORDER BY id DESC LIMIT 20", conn)
    #         st.write("Users (latest 10):", u)
    #         st.write("Sessions (latest 20):", s)
    #         total_users = pd.read_sql_query('SELECT COUNT(*) AS c FROM users', conn)['c'].iat[0]
    #         total_sessions = pd.read_sql_query('SELECT COUNT(*) AS c FROM sessions', conn)['c'].iat[0]
    #         st.info(f"Totals → users: {total_users}, sessions: {total_sessions}")
    #     except Exception as e:
    #         st.error(f"Debug error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__=="__main__":
    main()
