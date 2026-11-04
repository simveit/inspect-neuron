#!/usr/bin/env python3
"""Write a small, deterministic, license-safe corpus for smoke tests.

Every sentence below is synthetic and written for this repository; no third-party text is
included. The file is intentionally tiny: it exercises the harvest -> store -> score ->
report plumbing, not the science. Use a real corpus (a Hugging Face dataset or your own
JSONL/plain-text file) for a real experiment.

    python3 examples/make_tiny_corpus.py

writes examples/tiny-corpus.jsonl next to this script. The committed file is the output of
this script; rerun it after editing the sentences.
"""

import json
from pathlib import Path

SENTENCES = [
    "A pot of water came to a slow boil on the back burner.",
    "She chopped the onions finely and added them to the pan.",
    "The bread dough needs to rest for an hour before baking.",
    "He measured the flour twice and still had to add a little more.",
    "A pinch of salt brightens the taste of the soup.",
    "The kitchen smelled of rosemary, garlic, and warm butter.",
    "They picked the tomatoes while the morning was still cool.",
    "The garden gate creaked open in the wind.",
    "Rain has been falling on the seedlings since dawn.",
    "A single sunflower turned its head toward the light.",
    "The soil was dark and loose after the long rain.",
    "Bees moved slowly between the lavender and the mint.",
    "The compiler rejected the file because a bracket was missing.",
    "She wrote a small test before changing the parser.",
    "The build took longer than usual because the cache was cold.",
    "He read the error message twice and then checked the import.",
    "A clean commit makes the next review much easier.",
    "The function returns an empty list when there is nothing to do.",
    "Clouds gathered over the hills in the late afternoon.",
    "The forecast promised a clear and windy Saturday.",
    "A thin layer of frost covered the car window.",
    "The river ran high and brown after three days of rain.",
    "Sunlight broke through the blinds and crossed the floor.",
    "The evening air was cool enough for a jacket.",
    "She tuned the guitar before the rehearsal started.",
    "The drummer kept a steady beat through the whole set.",
    "A quiet melody drifted from the open window.",
    "He hummed the chorus while washing the dishes.",
    "The audience clapped for a long time after the last song.",
    "A piano stood alone in the corner of the empty hall.",
    "The train arrived four minutes late and nobody minded.",
    "They walked along the harbor and watched the boats return.",
    "The map showed a small path behind the old hotel.",
    "Her suitcase was packed a day before the flight.",
    "The bus stopped at every village along the coast.",
    "A paper ticket fell out of the guidebook.",
    "The cat slept on the warm windowsill all morning.",
    "A dog waited by the door with a muddy ball.",
    "The parrot repeated the kettle's whistle perfectly.",
    "He filled the bird feeder before the first snow.",
    "The rabbit disappeared into the tall grass.",
    "A goldfish circled the small pond twice and hid.",
    "The experiment produced a result that nobody expected.",
    "She recorded the temperature every hour in a notebook.",
    "The telescope was pointed at a distant cluster of stars.",
    "A simple model explained most of the pattern in the data.",
    "The students repeated the measurement and got the same number.",
    "He drew the process as boxes and arrows on the whiteboard.",
    "The library was quiet except for the turning of pages.",
    "A stack of letters sat unopened on the desk.",
    "The old clock in the hallway lost a minute every week.",
    "She found a pressed flower between the pages of a book.",
    "The market closed early because of the holiday.",
    "A handwritten sign offered apples and pears.",
    "The bakery sold out of bread before noon.",
    "He counted the change twice and put it in his pocket.",
    "A bicycle leaned against the fence, wet with rain.",
    "The streetlights came on while the sky was still pale.",
    "They shared an umbrella and walked the long way home.",
    "A neighbor waved from across the narrow street.",
]


def main() -> None:
    out = Path(__file__).with_name("tiny-corpus.jsonl")
    out.write_text("".join(json.dumps({"text": s}, ensure_ascii=False) + "\n" for s in SENTENCES))
    print(f"wrote {out} ({len(SENTENCES)} documents)")


if __name__ == "__main__":
    main()
