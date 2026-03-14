# Checklist Dat Muc Xuat Sac (20/20) - Mon NLP

Tai lieu nay doi chieu truc tiep voi rubric cham diem va chi ra minh chung can show trong luc bao ve.

## 1) Tong quan theo tieu chi

| Tieu chi | Muc tieu Xuat sac | Trang thai hien tai | Minh chung nen dua |
|---|---|---|---|
| Noi dung representation (4) | Slide truc quan, trinh bay mach lac, tra loi tot | Can chot storyline 10 phut + Q&A | `results/reports/NLP_project_report.tex`, phan **Demo Design for Defense**, phan **Hard-case** |
| Y nghia du an (4) | Van de thuc te + so lieu minh chung | Manh | Bang phan bo rating, metrics triage negative-first, issue extraction |
| Qua trinh lam viec (4) | Phan cong module, hieu biet cheo, commit/review cheo | Can bo sung minh chung quy trinh | phan **Team Workflow and Quality Assurance** trong report + anh commit/review tu repo git goc |
| Ket qua (4) | Quy trinh chuan, nhieu metrics, so sanh nhieu mo hinh, giai thich sai | Manh sau khi bo sung bang error case cu the | phan ket qua V0-V7, transformer, syllabus bench, bang `nlp_error_cases` |
| Demo (4) | Quy trinh chay on dinh, ket qua ro rang, de tai lap | Da co demo theo test + fallback CLI | `tests/`, `demo.py`, `demo_transformer.py` |

## 2) Cac diem can lam ngay truoc bao ve

1. Chot 8-10 slide theo timeline 10 phut o muc 3 ben duoi.
2. Chuan bi minh chung quy trinh nhom:
   - Screenshot commit graph.
   - 2-3 PR/review comments cheo.
   - 1 bang phan cong module theo thanh vien.
3. Chay thu demo theo test:
   - `python -m pytest tests/test_smoke_cli.py -q`
   - test 3 huong: positive, negative + issue, uncertain.
4. Chuan bi file backup demo CLI neu test gap loi:
   - `python demo.py "good but late delivery"`
   - `python demo_transformer.py "not bad"`

## 3) Storyline 10 phut (de dat diem representation)

1. 0:00-0:50 - Bai toan thuc te va KPI triage (khong bo sot review xau).
2. 0:50-2:00 - Du lieu, chia tap, class imbalance, vi sao dung recall_0/F2_0.
3. 2:00-3:40 - Pipeline NLP classic (normalization, negation, clause, char n-gram).
4. 3:40-5:20 - Ket qua sentiment: V0-V7 + threshold trade-off.
5. 5:20-6:40 - Issue extraction multi-label + metric tong quat/per-label.
6. 6:40-7:40 - Transformer + so sanh hard-case.
7. 7:40-8:30 - Error analysis ca sai cu the + cach giam rui ro.
8. 8:30-9:20 - Qua trinh nhom, review cheo, reproducibility.
9. 9:20-10:00 - Demo nhanh va ket luan.

## 4) Script demo ngan gon (de dat diem demo)

Input de chay:
- `great product, fast delivery`
- `card not working, support did not help`
- `idk`

Ky vong:
- Case 1 -> `POSITIVE`
- Case 2 -> `NEGATIVE` hoac `NEEDS_ATTENTION` + issue labels
- Case 3 -> `UNCERTAIN` + reason (`too_short`/`threshold_band`)

Trong luc demo, nhan manh:
- He thong khong ep du doan khi tin hieu yeu (safety).
- Issue extraction giup actionability (khong chi sentiment chung chung).

## 5) Q&A du kien (de dat diem tra loi cau hoi)

1. Tai sao khong dung accuracy lam metric chinh?
   - Vi du lieu lech lop 9.5:1; accuracy cao van co the bo sot lop negative.
2. Tai sao classic model van can khi da co transformer?
   - Classic cho recall negative cao va de control threshold/uncertainty.
3. Vi sao can multi-label issue extraction?
   - Mot review co the co nhieu van de dong thoi (delivery + redemption + support).
4. Cach tranh data leakage?
   - TF-IDF/Chi2 fit tren train-only, chon tham so tren val, seed co dinh, artifact luu ro.
5. Nhung case nao de sai?
   - Idiom/contrast phuc tap, text qua ngan, nhan hiem co it mau.

## 6) Luu y quan trong ve minh chung Git

Workspace hien tai khong con thu muc `.git`, vi vay khong trich duoc lich su commit/review truc tiep tu day.
De dat diem toi da o tieu chi "Qua trinh lam viec", nhom can mang them minh chung tu repo goc:

- Anh commit graph theo module.
- Anh review cheo (comment cua nguoi khong phai tac gia).
- Neu co: tag release truoc bao ve.
