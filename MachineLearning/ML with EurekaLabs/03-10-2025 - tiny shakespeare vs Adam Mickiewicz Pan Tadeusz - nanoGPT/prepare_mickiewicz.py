#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process a Polish text file according to the user's rules:
 - Remove lines that contain only digits and replace with a single blank line.
 - Remove bracketed footnote markers like [2], [53] -> replace with a single space.
 - Remove the character '«'.
 - Keep commas, colons, periods, and exclamation points.
 - Find/report all characters that are not part of the Polish alphabet (and not allowed punctuation or whitespace).
 - Save the processed text to disk.

Usage:
    python process_polish.py input.txt output.txt [--report non_polish_chars.txt]
"""

import argparse
import re
import sys

POLISH_LETTERS_LOWER = [
    'a','ą','b','c','ć','d','e','ę','f','g','h','i','j','k','l','ł','m','n','ń',
    'o','ó','p','r','s','ś','t','u','w','y','z','ź','ż'
]
# build set of allowed letters (lower and upper)
POLISH_LETTERS = set(POLISH_LETTERS_LOWER) | set([ch.upper() for ch in POLISH_LETTERS_LOWER])

# Allowed punctuation to be preserved
ALLOWED_PUNCT = {',', '.', '!', ':'}

# Pattern for bracketed footnotes like [2], [ 53 ] etc.
BRACKET_FOOTNOTE_RE = re.compile(r'\[\s*\d+\s*\]')

def process_text_char_by_char(text, non_polish_chars):
    """
    Walk text char-by-char; collect non-polish characters into non_polish_chars set.
    Returns text unchanged (processing of bracket removal and guillemet removal is done earlier).
    This function here mainly collects characters that are not in the allowed sets.
    """
    for ch in text:
        if ch in POLISH_LETTERS:
            continue
        if ch.isspace():
            continue
        if ch in ALLOWED_PUNCT:
            continue
        # If it's a digit, bracket, parentheses etc. we record it as non-polish char.
        non_polish_chars.add(ch)
    return text

def process_file(input_path, output_path, report_path=None):
    non_polish_chars = set()
    out_lines = []

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # If line contains only digits (maybe with surrounding whitespace/newline)
        if line.strip().isdigit():
            # replace with a single blank line (one newline)
            out_lines.append('\n')
            # record the digits as non-polish characters (if you want them reported)
            for ch in line:
                if ch.isdigit():
                    non_polish_chars.add(ch)
            continue

        # Remove guillemet '«' anywhere
        line = line.replace('«', '')
        line = line.replace('»', '')

        # Replace bracketed footnotes like [2], [53] with a single space
        # e.g. "Bramie[2]!" -> "Bramie !", we'll later keep punctuation next to word
        # To avoid double spaces we strip later if needed.
        line = BRACKET_FOOTNOTE_RE.sub(' ', line)

        # Optionally collapse multiple spaces produced by removals into single space,
        # but preserve leading indentation/newline behavior. We will collapse only inside line.
        # Keep trailing newline if it existed.
        has_newline = line.endswith('\n')
        core = line.rstrip('\n')
        # Replace runs of >=2 spaces with a single space to be tidy
        core = re.sub(r' {2,}', ' ', core)
        if has_newline:
            core = core + '\n'
        line = core

        # Scan char-by-char to collect non-polish characters (excluding allowed punct and whitespace)
        process_text_char_by_char(line, non_polish_chars)

        # Append processed line unchanged otherwise (we are not removing other punctuation except '«' and bracketed notes)
        out_lines.append(line)

    # Write processed output to disk
    with open(output_path, 'w', encoding='utf-8') as outf:
        outf.writelines(out_lines)

    # Prepare a sorted report
    sorted_chars = sorted(non_polish_chars, key=lambda c: (ord(c)))
    report_text = "Non-Polish characters found (excluding whitespace and allowed punctuation , . ! :):\n"
    if not sorted_chars:
        report_text += "  (none)\n"
    else:
        # Show hex code and repr for clarity
        for ch in sorted_chars:
            report_text += f"  U+{ord(ch):04X}  '{ch}'\n"

    # Print report to stdout
    print(report_text)

    # Optionally save report to file
    if report_path:
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write(report_text)
        print(f"Saved non-polish character report to: {report_path}")

    print(f"Processed text saved to: {output_path}")
    return non_polish_chars

def main(argv=None):
    parser = argparse.ArgumentParser(description="Process Polish text and report non-Polish characters.")
    parser.add_argument('input', help="Input text file path (UTF-8).")
    parser.add_argument('output', help="Output text file path to save processed text.")
    parser.add_argument('--report', help="Optional path to save a report of non-Polish characters.", default=None)
    args = parser.parse_args(argv)

    process_file(args.input, args.output, args.report)

if __name__ == '__main__':
    main()
