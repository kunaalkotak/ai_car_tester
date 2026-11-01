# Fresh AI challenge-image prompt bank

Use these as a controlled starting bank. To compare GPT, Gemini and Meta AI fairly, submit the same prompt text to each platform, using separate new sessions. Record the platform/model label, date, prompt ID and exact prompt in `data/prompt_log.csv`. Do not reuse prompts or generations that were in the V4 training/validation data. Keep original output files unchanged.

For each prompt, ask for one image only. Do not ask the model to add captions, labels, logos, watermarks, borders or collages. Do not keep regenerating until you get a preferred result without recording the rejected attempts; avoid selecting only images that look obviously synthetic. The purpose is a varied challenge set representative of likely listing photos.

## Suggested common prompt pattern

"Create one natural-looking photograph for a used-car online listing showing [vehicle description] from [camera angle] in [setting] under [lighting/weather]. Make the composition resemble an ordinary real-world car photograph. One car only. No text, captions, watermarks, graphic overlays or collage."

## 50 prompt variations

1. A white compact hatchback, front three-quarter view, parked in an apartment society parking area in bright afternoon daylight.
2. A silver midsize sedan, straight front view, parked beside a quiet suburban road on an overcast day.
3. A dark blue SUV, rear three-quarter view, parked in an open residential parking lot at dusk.
4. A red hatchback, left-side profile, parked outside a small independent car workshop.
5. A black luxury sedan, front three-quarter view, photographed in an underground parking garage with mixed lighting.
6. A grey compact SUV, rear view, parked beside a tree-lined road after light rain.
7. A beige family MPV, front view, parked in a driveway in soft morning light.
8. A yellow sports coupe, low front angle, parked on a paved roadside with buildings in the distance.
9. A white electric sedan, side profile, parked beside a charging station in daylight.
10. A green compact SUV, front three-quarter view, parked on a gravel driveway in cloudy weather.
11. A blue hatchback, rear three-quarter view, parked near a supermarket parking area.
12. A maroon sedan, side profile, parked beside a residential compound wall in midday sun.
13. A white pickup truck, front view, parked on a rural road with fields behind it.
14. A dark grey SUV, front three-quarter view, parked on a wet city street at night.
15. A silver hatchback, rear view, parked in a shaded apartment basement.
16. A red midsize sedan, front three-quarter view, photographed on a narrow urban street in morning light.
17. A navy blue SUV, side profile, parked on a beachside road under hazy daylight.
18. A pearl white sedan, rear three-quarter view, parked beneath trees with dappled sunlight.
19. A small white city car, front view, parked in a crowded but orderly public parking area.
20. A bronze crossover, side profile, parked outside a modern office building on an overcast day.
21. A black hatchback, front three-quarter view, parked on a sloped driveway in late afternoon light.
22. A blue sedan, rear view, parked near a highway rest stop in bright daylight.
23. A grey off-road SUV, front three-quarter view, parked on a muddy track after rain.
24. A white station wagon, side profile, parked beside a low brick building in soft daylight.
25. A red compact SUV, rear three-quarter view, parked near a row of shops at dusk.
26. A silver luxury coupe, front view, parked in a clean indoor showroom with ceiling lights.
27. A dark green sedan, side profile, parked on a quiet lane with parked vehicles in the background.
28. An orange compact hatchback, front three-quarter view, parked beside a sports ground.
29. A white minivan, rear view, parked in an outdoor family-home driveway in morning light.
30. A dark blue pickup, side profile, parked at a construction site entrance in cloudy daylight.
31. A grey sedan, close front detail showing the grille and headlights, parked outdoors.
32. A white SUV, close rear detail showing tailgate and taillights, parked outdoors.
33. A red hatchback, slightly distant full-car view, parked in a large open lot under harsh midday sun.
34. A black sedan, full-car view from a slightly elevated angle, parked in a residential street.
35. A blue crossover, front three-quarter view, photographed through light rain from a nearby sheltered area.
36. A champagne-coloured sedan, side profile, parked under warm streetlights after sunset.
37. A white compact SUV, front view, parked on a tree-lined road in foggy morning conditions.
38. A dark grey hatchback, rear three-quarter view, parked beside a concrete boundary wall.
39. A silver MPV, front three-quarter view, parked in an apartment visitor parking bay.
40. A deep red luxury sedan, side profile, parked in front of a hotel entrance in daylight.
41. A light blue small car, front three-quarter view, parked on a narrow residential lane.
42. A black SUV, rear view, parked on a gravel surface with hills in the background.
43. A white sedan, side profile, photographed in a dim covered parking space with a phone-camera look.
44. A grey hatchback, front three-quarter view, parked on a bright sunny street with strong shadows.
45. A dark blue electric SUV, rear three-quarter view, parked outside a contemporary home.
46. A silver compact car, front view, parked beside a petrol station in daytime.
47. A maroon SUV, side profile, parked near a public garden on an overcast day.
48. A white luxury coupe, rear three-quarter view, parked in an indoor parking structure.
49. A dark green compact hatchback, front three-quarter view, parked on a residential street after rainfall.
50. A bronze midsize SUV, full-car side view, parked beside a quiet road at golden hour.

## Suggested collection plan

- Run all 50 prompts once on each of the three platforms if possible: 150 AI images total.
- If a platform refuses a prompt or creates a non-car image, record it; do not silently replace it without logging.
- Record one prompt ID (e.g. `P001`) for matching outputs across all platforms. Give each provider a distinct group ID such as `GPT_P001`, `GEMINI_P001`, `META_P001`.
- For genuine photos, seek at least 100 independent camera-origin car photos across varied settings. Record the source/provenance and usage permission. Do not use AI-generated images as real controls.
- Check that challenge images are not already present in `data_v4_full`; the evaluation script performs an exact SHA-256 overlap check and excludes exact matches.
