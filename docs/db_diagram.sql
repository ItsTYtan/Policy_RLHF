Table "sections" {
  "section_title" VARCHAR(255) [pk]
  "date" TEXT 
  "text" TEXT 
    "summary" TEXT
}

Table "speeches" {
  "id" INT [pk, increment]
  "date" TEXT [not null]
  "speaker" TEXT [not null]
  "speech" TEXT [not null]
  "section_title" VARCHAR(255) [not null]
  "summary" TEXT

  Indexes {
    section_title [name: "idx_speeches_section"]
  }
}

Table "claims" {
  "id" INT [pk, increment]
  "claim" TEXT 
  "speech_id" INT

  Indexes {
    speech_id [name: "idx_claims_speech"]
  }
}

Table "dataset" {
  "id" INT [pk, increment]
  "topic" TEXT
  "subtopic" TEXT
  "question" TEXT
  "question_type" TEXT
  "generation" TEXT
  "model_name" TEXT
  "query_embedding" TEXT
}

Ref "fk_speeches_sections":"sections"."section_title" < "speeches"."section_title" [update: cascade, delete: cascade]

Ref "fk_claims_speeches":"speeches"."id" < "claims"."speech_id" [update: cascade, delete: cascade]