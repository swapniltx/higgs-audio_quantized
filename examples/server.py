from flask import Flask, request, jsonify
import base64
from boson_multimodal.generation import *

model_path = "bosonai/higgs-audio-v2-generation-3B-base"
audio_tokenizer = "bosonai/higgs-audio-v2-tokenizer"
max_new_tokens = 2048
Transcript = "Hello everyone, this is a test of the Higgs audio generation model."
ScenePrompt = "The audio is recorded in a calm and serene environment."
temperature = 0.7
top_k = 50
top_p = 0.95
ras_win_len = 72 
ras_win_max_num_repeat = 2
ref_audio_path = ".webcache/"
ref_audio = None 
ref_audio_in_system_message = False
chunk_method = None
chunk_max_word_num = 200
chunk_max_num_turns = 1
generation_chunk_buffer_size = None
seed = None
device_id = None
out_path = ".webcache/_output.wav"
use_static_kv_cache = 1
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# print(f"{CURR_DIR}/test.html")
# with open(f"{CURR_DIR}/test.html", "r", encoding="utf-8") as f:
#     print(f.read())
def TranscriptProsessing(transcript):
    transcript = normalize_chinese_punctuation(transcript)
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."
    return transcript

def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret

def prepare_generation_context(scene_prompt, ref_audio, ref_audio_path, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context for generation.

    The context contains the system message, user message, assistant message, and audio prompt if any.
    """
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split("|"))
        speaker_info_l = ref_audio.split("|")
        voice_profile = None
        if any([speaker_info.startswith("Profile:") for speaker_info in speaker_info_l]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("Profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("Profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(speaker_info_l):
            if character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name[8:].strip()}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name[8:].strip()}.txt")
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
        for spk_id, character_name in enumerate(speaker_info_l):
            if not character_name.startswith("profile:") and not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(ref_audio_path)
                # prompt_text_path = os.path.join(f"{CURR_DIR}/example/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                # assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                # with open(prompt_text_path, "r", encoding="utf-8") as f:
                prompt_text = character_name
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            # By default, we just alternate between male and female voices
            speaker_desc_l = []

            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids



if device_id is None:
    if torch.cuda.is_available():
        device_id = 0
        device = "cuda:0"
    else:
        device_id = None
        device = "cpu"
else:
    device = f"cuda:{device_id}"
audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=device)
model_client = HiggsAudioModelClient(
    model_path=model_path,
    audio_tokenizer=audio_tokenizer,
    device_id=device_id,
    max_new_tokens=max_new_tokens,
    use_static_kv_cache=use_static_kv_cache,
)
pattern = re.compile(r"\[(SPEAKER\d+)\]")

app = Flask(__name__)

# 允许所有来源的 CORS（仅开发用）
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/', methods=['GET'])
def index():
    # return app.send_static_file(f"{CURR_DIR}/test.html")
    with open(f"{CURR_DIR}/test.html", 'r', encoding="utf-8") as f:
        ret = f.read()
    return ret

# 处理 POST 请求
@app.route('/api', methods=['POST'])
def handle_post():
    data = request.json  # 获取 JSON 数据
    print("Received data:", data)
    save_path = ".webcache/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    transcript = data["Transcript"] or Transcript
    speaker_tags = sorted(set(pattern.findall(transcript)))
    transcript = TranscriptProsessing(transcript)
    scene_prompt = data["SencePrompt"] or ScenePrompt
    AudioTranscript = data["AudioTranscript"] or ref_audio
    # if AudioTranscript[:8] != "Profile:":
    with open(save_path + data["fileName"], 'wb') as f:
        f.write(base64.b64decode(data["fileBase64"]))

    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=AudioTranscript,
        ref_audio_path=save_path + data["fileName"],
        ref_audio_in_system_message=ref_audio_in_system_message,
        audio_tokenizer=audio_tokenizer,
        speaker_tags=speaker_tags,
    )
    chunked_text = prepare_chunk_text(
        transcript,
        chunk_method=chunk_method,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
    )

    logger.info("Chunks used for generation:")
    for idx, chunk_text in enumerate(chunked_text):
        logger.info(f"Chunk {idx}:")
        logger.info(chunk_text)
        logger.info("-----")

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=generation_chunk_buffer_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=seed,
    )

    sf.write(out_path, concat_wv, sr)
    logger.info(f"Wav file is saved to '{out_path}' with sample rate {sr}")
    with open(out_path, "rb") as f:
        audio_data = f.read()
    # return jsonify({
    # return "0"
    return jsonify({"audioData" :"data:audio/wav;base64,"+ base64.b64encode(audio_data).decode('utf-8'), })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)