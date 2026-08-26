# تشغيل النسخة المصححة

1. ارفع محتوى المشروع إلى GitHub.
2. شغل Streamlit من `app.py`.
3. افتح **Local Transformer Classifier** مباشرة.
4. اترك:
   - `Fetch URLs automatically before classification = ON`
   - `Maximum URLs = 2`
   - `Decision sensitivity = Balanced`
   - `Evidence chunks = 3`
5. اختر أول 1–3 حوادث فقط واضغط **Fetch + classify selected incidents**.
6. بعد التشغيل افحص أولاً قسم **Fetch status in this session**:
   - `FETCHED_DIRECT`
   - `FETCHED_JINA`
   - `FETCHED_WAYBACK`
   - أو `FAILED` مع سبب واضح.
7. افحص Status counts. يجب ألا تتحول كل النتائج إلى UNKNOWN بسبب semantic gate كما في النسخة السابقة.
8. لا تشغل 1207 حادثة قبل مراجعة عينة صغيرة يدوياً وضبط threshold إذا لزم.

ملاحظة: الـStatus Score هو NLI entailment score وليس probability calibrated علمياً.
