V7_ANTI_HALLUCINATION = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt to visualize the target. It may contain errors.
- Your goal is to refine the target image description for accurate image retrieval.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. Use it to CHECK whether the manipulation was applied correctly:
- Does it show the right color/shape/object changes?
- Does it preserve what should be preserved from the original?
- Does it add anything that the manipulation text did NOT ask for?

DO NOT describe what is in the Proxy Image. DO NOT add visual details from the Proxy Image (backgrounds, materials, textures, arrangements) into your description. The Proxy Image is AI-generated and contains hallucinated details.

## Guidelines on generating the Thoughts
    - First, what does the Original Image show? Focus on the KEY subject.
    - Second, what changes does the manipulation text request? Be precise.
    - Third, check the Proxy Image: did it apply the changes correctly? What errors exist?
    - Do NOT import new details from the Proxy Image into your reasoning.

## Guidelines on generating the Reflections
    - What did the Proxy get right or wrong about the manipulation intent?
    - What is the MINIMAL set of visual features that distinguish the target from similar images?

## Guidelines on generating Target Image Description
    - Describe ONLY the target image content — the key subject with the requested changes applied.
    - MUST be SHORT: similar length or shorter than the manipulation text.
    - Use only concrete visual attributes: type, color, pattern, shape.
    - NO backgrounds, NO environments, NO poses, NO artistic descriptions.
    - NO details that come from the Proxy Image rather than the manipulation text.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": <manipulation_text>
}

## Response
{
    "Original Image Description": <brief description of the key subject in the original>,
    "Thoughts": <manipulation analysis + proxy error check>,
    "Reflections": <key corrections needed>,
    "Target Image Description": <concise target description, MUST be short>
}
'''

V7_FASHIONIQ_GARMENT_ONLY = '''
- You are a fashion retrieval description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image**.
- **Image 2** is a **Proxy Image** generated from a first-round description. It may contain hallucinations.
- Your goal is to refine the target garment description for CLIP-based fashion retrieval.

## CRITICAL: How to use the Proxy Image
- The Proxy Image is ONLY a diagnostic tool.
- Use it to check whether the requested garment changes were applied correctly.
- DO NOT copy garment details from the proxy unless they are also justified by the manipulation text and the original image.
- DO NOT mention the model, pose, shoes, or background unless explicitly requested.

## Thoughts
    - Identify the original garment type and key attributes.
    - Identify the requested changes from the manipulation text.
    - Check whether the proxy applied those changes or introduced hallucinations.

## Reflections
    - Keep only the minimum garment attributes needed for retrieval.
    - Preserve unchanged garment attributes from the original when helpful.

## Target Image Description
    - MUST describe the garment only.
    - MUST be short: ideally 6-18 words.
    - Use only fashion attributes: type, color, sleeve length, neckline, pattern, material, silhouette, straps, length.
    - NO model, pose, shoes, background, or subjective styling language.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": <manipulation_text>
}

## Response
{
    "Original Image Description": <brief garment-focused description>,
    "Thoughts": <requested changes + proxy error check>,
    "Reflections": <minimum garment corrections needed>,
    "Target Image Description": <short garment-only target description>
}
'''

V7_FASHIONIQ_PRESERVE_UNCHANGED = '''
- You are a fashion retrieval description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image**.
- **Image 2** is a **Proxy Image** generated from a first-round description. It may contain hallucinations.
- Your goal is to refine the target garment description for CLIP-based fashion retrieval.

## CRITICAL
- Treat the Proxy Image only as a diagnostic tool.
- If the manipulation text does NOT explicitly request a change, preserve the corresponding garment attribute from the Original Image.
- Never let the Proxy Image overwrite unchanged attributes from the Original Image.

## Attribute Policy
- Requested attributes: apply the manipulation text exactly.
- Unrequested attributes: preserve from the Original Image when they are visually clear.
- If the Proxy conflicts with the Original on an unchanged attribute, trust the Original.

## Description Scope
- Describe only the garment.
- Prefer concrete fashion attributes: garment type, color, sleeve length, neckline, straps, pattern, material, silhouette, length.
- Keep the description short, but do not omit important unchanged attributes that help retrieval.
- No model, pose, shoes, background, or subjective style language.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": <manipulation_text>
}

## Response
{
    "Original Image Description": <brief garment-focused description>,
    "Thoughts": <requested changes + unchanged attributes to preserve + proxy error check>,
    "Reflections": <which attributes are changed and which are preserved>,
    "Target Image Description": <short garment-only target description with preserved unchanged attributes>
}
'''

V7_FOCUS = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt to visualize the target. It may contain errors.
- The manipulation text asks you to FOCUS on a specific object in the image. Your goal is to describe that object in enough detail for accurate image retrieval.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool:
- Does it correctly depict the focus object from the original?
- Does it preserve the object's real appearance (color, shape, material)?
- Does it add anything hallucinated that is NOT in the original image?

DO NOT import visual details from the Proxy Image into your description. It is AI-generated.

## Guidelines on generating the Thoughts
    - First, identify the focus object in the Original Image. What does it actually look like?
    - Second, note the object's key visual attributes: color, material, texture, shape, size, state.
    - Third, check the Proxy Image: did it faithfully reproduce the object's appearance?

## Guidelines on generating the Reflections
    - What are the REAL visual attributes of this object as seen in the Original Image?
    - What details would help a retrieval system distinguish this object from similar ones?

## Guidelines on generating Target Image Description
    - Describe the focus object with SPECIFIC visual details from the Original Image.
    - Include: object type, color, material/texture, shape, notable features.
    - Keep it concise but DESCRIPTIVE — at least 8-15 words. Do NOT reduce to just the object name.
    - Focus on the OBJECT itself, not the surrounding scene or background.
    - NO details that come from the Proxy Image rather than the Original Image.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": <manipulation_text>
}

## Response
{
    "Original Image Description": <brief description of the focus object as seen in the original>,
    "Thoughts": <object analysis + proxy error check>,
    "Reflections": <key visual attributes to preserve>,
    "Target Image Description": <descriptive target, 8-15 words, with real visual details>
}
'''

GENECIS_CHANGE_OBJECT = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt. It may contain errors.
- The manipulation text names an OBJECT to change. The target is the same scene with that object different.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. DO NOT describe it or copy its details.

## Response Rules
- Target Description MUST be 5-12 words.
- Name the changed object with ONE specific visual attribute (color, material, or type).
- Add ONE unchanged context element to anchor the scene.
- DO NOT speculate about what the change looks like. Describe what STAYS the same plus the object name.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": "<manipulation_text>"
}

## Response
{
    "Thoughts": <what is the named object in original, what context stays>,
    "Target Image Description": <5-12 words: changed object + one scene anchor>
}
'''

GENECIS_CHANGE_ATTRIBUTE = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt. It may contain errors.
- The manipulation text names an ATTRIBUTE. The target is the same scene with that attribute changed.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. DO NOT describe it or copy its details.

## Response Rules
- Target Description MUST be 5-12 words.
- State the key subject from the original, then the attribute with its CHANGED value.
- Be SPECIFIC about attribute values: exact color names, textures, materials.
- DO NOT add background, environment, or artistic details.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": "<manipulation_text>"
}

## Response
{
    "Thoughts": <what attribute changes, what is the main subject>,
    "Target Image Description": <5-12 words: subject + changed attribute value>
}
'''

GENECIS_FOCUS_OBJECT = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt. It may contain errors.
- The manipulation text names an OBJECT to focus on. Describe that object with enough detail for retrieval.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. DO NOT describe it or copy its details.

## Response Rules
- Target Description MUST be 8-15 words.
- Describe the named object with SPECIFIC visual details FROM THE ORIGINAL: color, material, texture, shape, condition.
- Do NOT just name the object. Add 2-3 concrete visual attributes you actually SEE.
- Focus on the OBJECT only, not the scene.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": "<manipulation_text>"
}

## Response
{
    "Thoughts": <real visual attributes of the object in the original>,
    "Target Image Description": <8-15 words: object with specific visual details>
}
'''

GENECIS_FOCUS_ATTRIBUTE = '''
- You are an image description expert. You are given TWO images and a manipulation text.
- **Image 1** is the **Original Image** (the reference).
- **Image 2** is a **Proxy Image** — an AI-generated attempt. It may contain errors.
- The manipulation text names an ATTRIBUTE to focus on. Describe the scene emphasizing that attribute.

## CRITICAL: How to use the Proxy Image
The Proxy Image is ONLY a diagnostic tool. DO NOT describe it or copy its details.

## Response Rules
- Target Description MUST be 8-15 words.
- Name the 2-3 most distinctive values of the attribute FROM THE ORIGINAL and their objects.
- For "color": list specific color names and what has each color.
- Be concrete: "red brick, green sign, blue sky" not "colorful scene".
- NO backgrounds, NO artistic descriptions.

## Input
{
    "Original Image": <image_1>,
    "Proxy Image": <image_2>,
    "Manipulation text": "<manipulation_text>"
}

## Response
{
    "Thoughts": <specific attribute values observed in original>,
    "Target Image Description": <8-15 words: key attribute values and their objects>
}
'''

PROMPT_VARIANTS = {
    'v7_anti_hallucination': V7_ANTI_HALLUCINATION,
    'v7_fashioniq_garment_only': V7_FASHIONIQ_GARMENT_ONLY,
    'v7_fashioniq_preserve_unchanged': V7_FASHIONIQ_PRESERVE_UNCHANGED,
    'v7_focus': V7_FOCUS,
    'genecis_change_object': GENECIS_CHANGE_OBJECT,
    'genecis_change_attribute': GENECIS_CHANGE_ATTRIBUTE,
    'genecis_focus_object': GENECIS_FOCUS_OBJECT,
    'genecis_focus_attribute': GENECIS_FOCUS_ATTRIBUTE,
}
