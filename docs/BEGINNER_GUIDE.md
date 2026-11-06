# A beginner's guide to the chartqa-vlm-capstone-project repo

Repo: https://github.com/parulrewatkar0692/chartqa-vlm-capstone-project

## 1. What this project is

We want to teach an AI model to answer questions about charts, such as "What is the value of the dark blue bar in 2019?" or "What is the average of the USA and France bars?".

You teach a model the way you teach a student: you show it many worked examples, each a question with the correct answer and a short explanation. This is called **fine-tuning**. This repo holds those worked examples (the **dataset**), the chart pictures they refer to, and a notebook that tests a model *before* any teaching, to give a starting score (the **baseline**).

The model we use is a **vision-language model (VLM)**: it can look at a picture, not only read text. So each example shows the model the chart *image* together with the question, and the model has to read the chart itself. For an easier version, each example can also include the chart's data as text.

Here is one chart from the dataset:

![A bar chart of World Cup prize money](../data/v1.0/images/train/two_col_315.png)

The question for this chart is "In the 1990s, what is the average prize money (in millions) in the USA and France?". The right answer is 87, because 1994 (USA) is 71, 1998 (France) is 103, and (71 + 103) / 2 = 87.

## 2. The flow, step by step

Think of it as a kitchen: raw ingredients come in, get cleaned and sorted, are checked by a head chef, and only the good plates go out.

**Step 1. Raw ingredients: the ChartQA dataset.**
ChartQA is a public research dataset of about 21,000 charts, each with questions written by people. Each chart comes with its picture, its data table and a description file (title, chart type, colours). The full download is large (about 875 MB), so this repo keeps only the pictures it uses (about 70 MB).

**Step 2. Keep only human-written questions.**
ChartQA also has machine-generated questions, which are easier and repetitive. We keep only the human-written ones (about 9,600).

**Step 3. Turn each chart's data into text.**
For every chart we write out its source, chart type, title, which colour belongs to which series (for example "Male = dark blue"), and the data table. This text is used by the teacher in step 7 and to check its work in step 8. In the easier `image_table` version, the model sees it too.

**Step 4. Throw out questions that can't be answered from that text.**
Some questions ask about things the data doesn't show, such as gridlines, fonts or dotted lines. Some refer to a colour that doesn't exist in that chart. Some have an answer that doesn't appear in the data. All of these are removed.

**Step 5. Keep the business, economy and health topics.**
A keyword filter keeps charts about money, the economy, health and business, to match the project's theme.

**Step 6. Split into three sets.**
- **Train** (1,403 examples): the examples the model learns from.
- **Validation** (167): used during training to check progress and pick the best version.
- **Test** (180): locked away and used only once, for the final score. The model never sees these while learning, like an exam paper kept sealed.

No chart appears in more than one set, so the model can't simply memorise a chart and recognise it in the exam. The sets are also balanced, so each one has a similar mix of sources (Statista, Pew, Our World in Data, OECD), chart types (bar, line, pie) and question types (look-up, arithmetic, colour-based).

About 5% of the questions are deliberately impossible: a question paired with the wrong chart. The correct answer is "I can't answer this from the chart". This teaches the model to admit when it doesn't know instead of guessing.

**Step 7. Add "show your working" with a teacher model.**
The original dataset gives only the final answer. A large AI model, **Claude Opus**, acted as the teacher. For each training example it was shown the chart's data text, the question and the correct answer, and it wrote:
- **evidence**: exactly which cells of the table (or which colour, or the title) the answer comes from;
- **rationale**: one or two sentences of reasoning, e.g. "(71 + 103) / 2 = 87".

The teacher was never allowed to change the answer.

**Step 8. Check the teacher's work automatically.**
A checking script (`scripts/check_teacher.py`) confirms that every piece of evidence really exists in the chart data, that the answer matches the original, and that the format is correct. When the teacher said "this original answer is wrong", which happened for about 10% of examples (mostly charts with messy tables), the example was removed and replaced with a similar one that passed every check.

**Step 9. Attach the chart picture.**
Each example now points to its chart image (a PNG file). The question the model receives is *the picture plus the question*.

**Step 10. Check every picture with the teacher.**
Steps 7 and 8 used the data text, not the picture. So Claude Opus then looked at every train and validation **picture** and answered two things: can this be answered from the picture alone, and does the stored answer match what the picture shows? This removed 46 training and 3 validation examples. Most (16 of the 20 wrong answers) were percentages stored as fractions, like 0.64 for a chart that shows 64%. Each remaining example is marked `yes` (values clearly printed), `approx` (you must estimate from the axis) or `no` (the correct answer is "can't answer") in `flags.image_readable`. The test set was not checked this way.

**Step 11. Human spot-check (still to do).**
Two team members independently review 150 random training examples and mark whether each is correct. Their agreement is measured. This is the final quality check.

**Step 12. Freeze.**
After the human check, the dataset is "frozen" (tagged `v1.0-frozen` in the repo). From then on nobody changes it, so every experiment uses the same data and the results can be compared fairly.

**Step 13. Baseline test.**
The notebook runs a ready-made vision model (for example Qwen2.5-VL-7B) on the 180 test questions *before* any fine-tuning and scores it. Later, the fine-tuned model is scored the same way to show how much it improved.

## 3. What each folder and file contains

```
chartqa-vlm-capstone-project/
├── README.md                  ← start here: short overview and one example record
├── docs/
│   └── BEGINNER_GUIDE.md      ← this guide
├── data/
│   └── v1.0/                  ← ★ THE DATASET ★
│       ├── image_only/        ← main setting: chart picture + question
│       │   ├── train.jsonl    ← 1,403 training examples with evidence and reasoning
│       │   ├── val.jsonl      ← 167 validation examples
│       │   ├── test.jsonl     ← 180 sealed exam questions (never train on these)
│       │   └── answer_only/   ← same examples without the reasoning, for a comparison experiment
│       ├── image_table/       ← easier setting: picture + the chart's data as text + question
│       │   ├── train.jsonl, val.jsonl, test.jsonl
│       │   └── answer_only/
│       ├── images/            ← the chart pictures, one PNG per chart
│       │   ├── train/         ← 1,104 pictures
│       │   ├── val/           ← 140 pictures
│       │   └── test/          ← 157 pictures
│       ├── stats.json         ← counts per set, picture sizes, results of the picture check
│       ├── teacher_stats.json ← how many examples the teacher accepted, rejected and replaced
│       └── MANIFEST.txt       ← fingerprints (checksums) of each file, to prove nothing changed
├── scripts/
│   └── check_teacher.py       ← the automatic checker from step 8
└── notebooks/
    └── 01_vlm_baselines.ipynb ← Google Colab notebook: runs a ready-made vision model on the test set and scores it
```

**About `.jsonl` files:** each line is one example, written in JSON (a simple "label: value" text format). You can open them in any text editor.

**About `.py` files:** `check_teacher.py` is a Python script. You don't need it to *use* the dataset, only to re-check evidence.

**Which files to pick:** train on `image_only/train.jsonl` (or `image_table/train.jsonl` for the easier version). Pictures are not inside the `.jsonl` files; each line has an `image` field with the picture's path, relative to `data/v1.0/`.

## 4. What one example looks like

Each line in `image_only/train.jsonl` is a short conversation in three parts. This is the example for the chart above:

1. **System**: the standing instruction. "You are a chart analyst. Answer questions about the chart image provided. Return only valid JSON…"
2. **User**: the chart picture plus the question:
   ```
   [image: images/train/two_col_315.png]
   Question: In the 1990s, what is the average prize money (in millions) in the USA and France?
   Respond with JSON: {"answer": ..., "answer_type": ..., "unit": ..., "evidence": [...], "rationale": ...}
   ```
3. **Assistant**: the model answer we want it to learn:
   ```json
   {"answer": 87, "answer_type": "number", "unit": "million U.S. dollars",
    "evidence": ["Row=1994 (USA), Col=Prize money in million U.S. dollars (71.0)",
                 "Row=1998 (France), Col=Prize money in million U.S. dollars (103.0)"],
    "rationale": "USA 1994 was 71 and France 1998 was 103; (71 + 103) / 2 = 87."}
   ```

Read `Row=1994 (USA), Col=Prize money… (71.0)` as "the bar labelled 1994 (USA), in the Prize money series, whose value is 71.0". Other evidence forms are `Colour: Maximum price = dark blue` and `Title`.

Each line also carries labels that are not shown to the model, which we use to break down scores:

| Label | Example | Meaning |
|---|---|---|
| `id`, `chart_id` | `train-two_col_315-3611` | The example's unique name, and which chart it is about |
| `source` | `statista` | Where the chart came from: `statista`, `pew`, `owid`, `oecd` |
| `chart_type` | `v_bar` | `v_bar` vertical bars, `h_bar` horizontal bars, `line`, `pie` |
| `answer_type` | `number` | `number`, `text`, `yes_no`, `unanswerable` |
| `question_type` | `arithmetic_comparison` | `retrieval` (read one value), `arithmetic_comparison`, `visual_reference` (colours, shapes) |
| `target_status` | `teacher_verified` | `teacher_verified`, or `fixed_refusal` for the can't-answer examples |
| `flags.image_readable` | `yes` | Result of the picture check (step 10) |
| `input_text` | `Source: Statista ...` | The chart's data as text, kept so evidence can be checked |

A can't-answer example looks the same, but its picture belongs to a different chart than the question was written for, and the answer the model should give is `{"answer": null, "answer_type": "unanswerable", ...}`.

## 5. How to use it

- **To fine-tune:** train on `data/v1.0/image_only/train.jsonl`, check progress on `val.jsonl`, and report the final score on `test.jsonl` only once.
- **To compare with and without reasoning:** train one model on the full examples and one on `answer_only/`. If the first scores higher, teaching the model to show evidence and reasoning helped.
- **To get the baseline score:** open `notebooks/01_vlm_baselines.ipynb` in Google Colab, choose an L4 or A100 GPU and follow the notes at the top of the notebook.
- **To understand the numbers:** read `data/v1.0/stats.json` and `teacher_stats.json`.

## 6. Still to do

- Human review of 150 training examples by two people, then tag the repo `v1.0-frozen`.
- Run the baseline notebook for each base model and record the scores.
- Check the 180 test answers against their pictures too. The test set was not part of step 10 and probably has a few percentages stored as fractions, so look at percentage questions first when reading test errors.
- Write the fine-tuning notebook.

## 7. Small glossary

| Term | Meaning |
|---|---|
| Dataset | The collection of worked examples the model learns from |
| VLM (vision-language model) | An AI model that reads pictures as well as text, e.g. Qwen2.5-VL |
| Fine-tuning | Teaching an existing model with your own examples |
| QLoRA | A cheap way to fine-tune: the big model stays frozen and compressed, and only small add-on layers are trained, so it fits on one GPU |
| Base model | The ready-made model before fine-tuning |
| Baseline | The base model's score before fine-tuning: the "before" photo |
| Train / validation / test | Learn from / check progress on / final sealed exam |
| Gold answer | The correct answer from the original dataset |
| Teacher model | A bigger AI (Claude Opus) that wrote the explanations |
| Evidence | The exact cells, colours or title an answer is based on ("grounding") |
| Rationale | A one- or two-sentence explanation of how the answer was reached |
| Unanswerable example | A question the chart can't answer; the right reply is `null` |
| Stratified | Each set keeps the same mix of chart and question types |
| Relaxed accuracy | A number counts as correct if it's within 5% of the gold answer |
| JSON / JSONL | A text format for structured data / a file with one JSON example per line |
| Percent vs fraction mistake | A gold answer stored as 0.64 when the chart shows 64% |
| Freeze | Lock the dataset so nobody changes it during experiments |
