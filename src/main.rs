use eliza_lab::{ElizaEngine, MAX_INPUT_CHARS};
use std::env;
use std::io::{self, BufRead, Write};

const MAX_INPUT_BYTES: usize = MAX_INPUT_CHARS * 4;

enum InputLine {
    Text(String),
    TooLong,
}

fn read_bounded_line(reader: &mut impl BufRead) -> io::Result<Option<InputLine>> {
    let mut bytes = Vec::with_capacity(MAX_INPUT_BYTES.min(256));
    let mut discarded = false;
    let mut saw_input = false;

    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            if !saw_input {
                return Ok(None);
            }
            break;
        }
        saw_input = true;
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let content_length = newline.unwrap_or(buffer.len());
        let remaining = (MAX_INPUT_BYTES + 2).saturating_sub(bytes.len());
        let copy_length = content_length.min(remaining);
        bytes.extend_from_slice(&buffer[..copy_length]);
        discarded |= content_length > remaining;

        let consumed = newline.map_or(buffer.len(), |position| position + 1);
        reader.consume(consumed);
        if newline.is_some() {
            break;
        }
    }

    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    if discarded || bytes.len() > MAX_INPUT_BYTES {
        return Ok(Some(InputLine::TooLong));
    }
    let text = String::from_utf8(bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(Some(InputLine::Text(text)))
}

fn main() -> io::Result<()> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().map(String::as_str) == Some("--once") {
        let input = arguments
            .iter()
            .skip(1)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let mut engine = ElizaEngine::new();
        print_reply(&engine.respond(&input));
        return Ok(());
    }

    println!("ELIZA Lab — a local rule-based dialogue experiment");
    println!("Educational software, not therapy or medical advice. No transcript is stored.");
    println!("Type /quit to leave.\n");

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut engine = ElizaEngine::new();
    let mut stdout = io::stdout().lock();
    write!(stdout, "you > ")?;
    stdout.flush()?;

    while let Some(line) = read_bounded_line(&mut stdin)? {
        let input = match line {
            InputLine::Text(input) => input,
            InputLine::TooLong => "x".repeat(MAX_INPUT_CHARS + 1),
        };
        if matches!(input.trim(), "/quit" | "/exit") {
            writeln!(stdout, "eliza > Goodbye.")?;
            break;
        }

        let response = engine.respond(&input);
        writeln!(stdout, "eliza > {}", response.text)?;
        writeln!(
            stdout,
            "trace > {} (turn {})",
            response.rule_id, response.turn
        )?;
        write!(stdout, "you > ")?;
        stdout.flush()?;
    }

    Ok(())
}

fn print_reply(reply: &eliza_lab::Reply) {
    println!("{}", reply.text);
    println!("rule={} turn={}", reply.rule_id, reply.turn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn bounded_reader_drains_an_oversized_line_before_the_next_prompt() {
        let input = format!("{}\nnext\n", "x".repeat(MAX_INPUT_BYTES + 20));
        let mut reader = Cursor::new(input.into_bytes());

        assert!(matches!(
            read_bounded_line(&mut reader).unwrap(),
            Some(InputLine::TooLong)
        ));
        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value, "next"),
            _ => panic!("the next bounded line should remain readable"),
        }
    }

    #[test]
    fn bounded_reader_accepts_a_maximum_size_utf8_line() {
        let input = format!("{}\r\n", "🙂".repeat(MAX_INPUT_CHARS));
        let mut reader = Cursor::new(input.into_bytes());

        match read_bounded_line(&mut reader).unwrap() {
            Some(InputLine::Text(value)) => assert_eq!(value.chars().count(), MAX_INPUT_CHARS),
            _ => panic!("a maximum-size UTF-8 line should be accepted"),
        }
    }
}
