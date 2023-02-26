use eliza::Eliza;
use mysql::prelude::*;
use mysql::*;
use std::sync::{Arc, Mutex};
use teloxide::dptree::di::Asyncify;
use teloxide::prelude::*;
use tokio;
use tokio::spawn;

#[tokio::main]
async fn main() {
    let bot = Bot::from_env();
    setup_db(&mut establish_connection(DatabaseCredentials::default()));
    Dispatcher::builder(bot, Update::filter_message().endpoint(Asyncify(answer)))
        .dependencies(dptree::deps![Arc::new(Mutex::new(
            Eliza::from_file("config/doctor.json").unwrap()
        ))])
        .build()
        .dispatch()
        .await;
}

fn answer(bot: Bot, msg: Message, eliza_container: Arc<Mutex<Eliza>>) -> ResponseResult<()> {
    let mut eliza = eliza_container.lock().unwrap();
    let response = match msg.text() {
        Some(text) => match text {
            "/start" => eliza.greet(),
            "/diagnosis" => eliza.farewell(),
            _ => eliza.respond(text),
        },
        None => "not valid message".to_string(),
    };
    put_chat(
        &mut establish_connection(DatabaseCredentials::default()),
        msg.from().unwrap().username.as_ref().unwrap(),
        msg.from().unwrap().id.0,
        msg.text().unwrap(),
        &response,
    );

    spawn(async move {
        bot.send_message(msg.chat.id, response).await;
    });
    Ok(())
}

fn setup_db(db_connection: &mut Conn) {
    db_connection
        .query_drop("CREATE DATABASE IF NOT EXISTS TBot")
        .unwrap();
    db_connection
        .query_drop(
            "CREATE TABLE IF NOT EXISTS TBot.Chats (
        id INT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
        user_name VARCHAR(35) NOT NULL,
        user_id int(15) UNSIGNED NOT NULL,
        prompt MEDIUMTEXT NOT NULL, 
        response MEDIUMTEXT NOT NULL, 
        msgtime datetime default now() NOT NULL)",
        )
        .unwrap();
    db_connection
        .query_drop(
            "CREATE TABLE IF NOT EXISTS TBot.Admin (
        user_name VARCHAR(35) PRIMARY KEY,
        password VARCHAR(50) NOT NULL)",
        )
        .unwrap();
}

struct DatabaseCredentials {
    user: String,
    password: String,
    host: String,
    tcp_port: u16,
}

impl DatabaseCredentials {
    fn new(user: String, password: String, host: String, tcp_port: u16) -> DatabaseCredentials {
        DatabaseCredentials {
            user: (user),
            password: (password),
            host: (host),
            tcp_port: (tcp_port),
        }
    }
    fn default() -> DatabaseCredentials {
        DatabaseCredentials {
            user: ("root".to_string()),
            password: ("".to_string()),
            host: ("localhost".to_string()),
            tcp_port: (3306),
        }
    }
}

fn establish_connection(credentials: DatabaseCredentials) -> Conn {
    let connection = Conn::new(
        OptsBuilder::new()
            .user(Some(credentials.user))
            .pass(Some(credentials.password))
            .ip_or_hostname(Some(credentials.host))
            .tcp_port(credentials.tcp_port),
    )
    .unwrap();

    return connection;
}

fn put_chat(
    db_connection: &mut Conn,
    user_name: &str,
    user_id: u64,
    prompt: &str,
    response: &String,
) {
    db_connection
        .exec_drop(
            "INSERT INTO TBot.Chats (user_name, user_id, prompt, response)
    VALUES (:user_name, :user_id, :prompt, :response)",
            params! {
                user_name,
                user_id,
                prompt,
                response
            },
        )
        .unwrap();
}
