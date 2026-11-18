# Field checklist — one page, take this with you

## Before leaving

- [ ] IEC approval letter (carry a copy)
- [ ] Printed forms, **100% scale, A4, not "fit to page"**
- [ ] Participant information sheets (one per prescriber)
- [ ] Consent forms (one per prescriber)
- [ ] Pens (and let people use their own if they prefer)
- [ ] Phone charged; storage free
- [ ] The next unused writer code written down

## Per prescriber (~6 min)

1. [ ] Give information sheet, answer questions
2. [ ] **Consent signed before any writing**
3. [ ] Assign writer code → write it on the consent form **and** in the form's
       "Writer code" field
4. [ ] Hand over 2 sheets. Say: *"normal speed, normal handwriting — please
       don't write neatly for us"*
5. [ ] Check: **no patient information anywhere on the sheet**
6. [ ] Photograph: whole sheet, all four corner squares visible, no flash, no
       shadow
7. [ ] Rename files `W03_LRG-001.jpg`
8. [ ] Run ingest **now** if you can — re-shoot anything skipped before they
       leave

```bash
python scripts/ingest_forms.py --photos today/ --out data/real
```

## End of day

- [ ] Every photo ingested or re-shot
- [ ] Consent forms filed separately from the images
- [ ] Writer-code list updated
- [ ] Sheets stored or destroyed per your IEC ruling

## If something goes wrong

| Symptom | Fix |
|---|---|
| "found markers [0, 1, 3]" | A corner was cut off or in glare. Re-shoot with the whole sheet in frame. |
| Many cells reported blank | Photo too dark or out of focus. Re-shoot in better light. |
| Labels look wrong on spot-check | Forms were printed scaled. Reprint at 100% and re-collect. |
| Prescriber wrote a patient name | Do not photograph. Destroy the sheet. Issue a fresh one. |
