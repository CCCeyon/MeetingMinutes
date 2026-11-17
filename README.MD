# FS Transcriber

### Step 1: Setup the Project
Setup your python-venv using python 3.13 (other versions may work)
and run
```bash
pip install -r requirements.txt
```
Run
```bash
cp .env.example .env
```
and fill in your HuggingFace-Token, to get access to the pyannote/speaker-diarization-community-1 model.

Run
```bash
cp config/members.example.txt config/members.txt
```
and fill in the full names of your members

### Step 2: Upload your Recording
Upload your Recording to the `recordings` folder. Most Audio-Formats are supported.

### Step 3: Transcribe the Recording
Run 
```bash
python3 transcribe.py recordings/meeting.m4a
```
This may take a while for the first run, as it downloads multiple models from huggingface.

The transcript and the minutes are saved to `results/<recording>_transcript.txt` and
`results/<recording>_minutes.txt`.

#### Arguments

| Argument | Description | Default |
|---|---|---|
| `audio` | Path to the recording (required) | – |
| `-l`, `--language LANGUAGE` | Language of the minutes | `English` |
| `-n`, `--name LABEL=NAME` | Rename a speaker label, e.g. `-n SPEAKER_00=Anna`. Can be used multiple times | – |
| `-m`, `--members MEMBERS` | File with the full names of your members, one per line | `config/members.txt` |
| `-o`, `--output-dir OUTPUT_DIR` | Folder for the transcript and minutes | `results` |
| `-h`, `--help` | Show all arguments and exit | – |

Example with all optional arguments:
```bash
python3 transcribe.py recordings/meeting.m4a --language German -n SPEAKER_00=Anna -n SPEAKER_01=Ben --members config/members.txt --output-dir results
```

The prompt for the minutes can be adjusted in `config/prompt.txt`.