"""Download the public-domain novel for lesson 09; standard library only.

Run: python Seminars/transformers/download_captains_daughter.py
No model code, training, or tests are executed. Existing output files are never overwritten.
"""
import hashlib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


TITLE = 'Капитанская_дочка_(Пушкин)/1978_(СО)'
URL = 'https://ru.wikisource.org/wiki/' + urllib.parse.quote(TITLE)
DESTINATION = Path(__file__).resolve().parent / 'data' / 'captains_daughter'
ROMANS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV']


class ParagraphReader(HTMLParser):
    """Keep paragraph text and line breaks; drop superscript footnote markers."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = False
        self.skip = 0
        self.parts = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag in ('sup', 'script', 'style'):
            self.skip += 1
        if tag == 'p':
            self.inside = True
            self.parts = []
        elif tag == 'br' and self.inside and not self.skip:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('sup', 'script', 'style'):
            self.skip = max(0, self.skip - 1)
        if tag == 'p' and self.inside:
            text = unicodedata.normalize('NFC', ''.join(self.parts))
            text = '\n'.join(re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.splitlines()).strip()
            if text:
                self.paragraphs.append(text)
            self.inside = False

    def handle_data(self, data):
        if self.inside and not self.skip:
            self.parts.append(data)


def main():
    names = ['source.html', 'chapters.json', 'captains_daughter.txt', 'source.json']
    if any((DESTINATION / name).exists() for name in names):
        raise FileExistsError(f'Dataset already exists in {DESTINATION}; reuse it, do not overwrite.')
    # Pin the source revision used when the lesson was prepared.
    request = urllib.request.Request('https://ru.wikisource.org/w/index.php?oldid=5678873', headers={
        'User-Agent': 'DeepSchoolLearning/1.0 (personal educational text download)',
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
    html = raw.decode('utf-8')
    headings = list(re.finditer(r'<h3\b[^>]*>(.*?)</h3>', html, re.DOTALL | re.IGNORECASE))
    chapters = []
    for index, heading in enumerate(headings):
        title_reader = ParagraphReader()
        title_reader.feed('<p>' + heading.group(1) + '</p>')
        title = ' '.join(title_reader.paragraphs)
        match = re.match(r'^глава\s+([IVX]+)\.', title, re.IGNORECASE)
        if not match:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(html)
        reader = ParagraphReader()
        reader.feed(html[heading.end():end])
        text = '\n\n'.join(reader.paragraphs)
        if len(text) < 1000:
            raise ValueError(f'Unexpectedly short chapter: {title}; inspect source HTML.')
        chapters.append({'number': match.group(1).upper(), 'title': title, 'text': text})
    if [chapter['number'] for chapter in chapters] != ROMANS:
        raise ValueError('Expected exactly chapters I–XIV in order; source markup may have changed.')
    revision = re.search(r'"wgRevisionId"\s*:\s*(\d+)', html)
    revision_id = int(revision.group(1)) if revision else None
    text = '\n\n'.join(chapter['title'] + '\n\n' + chapter['text'] for chapter in chapters) + '\n'
    metadata = {
        'author': 'Александр Сергеевич Пушкин',
        'title': 'Капитанская дочка',
        'source_url': URL,
        'revision_id': revision_id,
        'revision_url': f'https://ru.wikisource.org/w/index.php?oldid={revision_id}' if revision_id else None,
        'downloaded_utc': datetime.now(timezone.utc).isoformat(),
        'html_sha256': hashlib.sha256(raw).hexdigest(),
        'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'rights_note': 'Wikisource labels the original work public domain in Russia and life+70 jurisdictions. Editorial appendices and notes excluded.',
        'extraction': 'Paragraphs in chapters I–XIV, including epigraphs. HTML navigation, superscript markers, post-chapter editorial material and appendix excluded. NFC, whitespace normalization; original case preserved.',
        'chapter_characters': {c['number']: len(c['text']) for c in chapters},
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'source.html').write_bytes(raw)
    (DESTINATION / 'chapters.json').write_text(json.dumps(chapters, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (DESTINATION / 'captains_daughter.txt').write_text(text, encoding='utf-8')
    (DESTINATION / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'directory': str(DESTINATION), 'chapters': len(chapters),
                      'characters': len(text), 'revision_id': revision_id,
                      'first_paragraph': chapters[0]['text'][:500],
                      'last_paragraph': chapters[-1]['text'][-500:]}, ensure_ascii=True))


if __name__ == '__main__':
    main()
