use eliza_lab::ElizaEngine;
use std::env;
use std::io::{self, BufRead, Write};

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
    let mut engine = ElizaEngine::new();
    let mut stdout = io::stdout().lock();
    write!(stdout, "you > ")?;
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let input = line?;
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
