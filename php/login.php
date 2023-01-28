<?php
session_start();
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/@picocss/pico@1.*/css/pico.classless.min.css">
    <title>Login</title>
</head>

<body>
    <main>
        <img src="assets/bot-icon.png" alt="icon" style="width:30%;margin-left:33%;margin-right:33%;" />
        <form method="POST">
            <label for="username"><b>Nome utente</b></label>
            <input type="text" name="username" placeholder="Inserisci il tuo nome utente" max="8">
            <br>
            <label for="password"><b>Password</b></label>
            <input type="password" name="password" placeholder="Inserisci la tua password" max="8">
            <br>
            <input type="submit" name="login" value="Accedi" />
        </form>
        <?php
        if (isset($_POST["login"])) {
            try {
                $connection = new mysqli("localhost", "root", "", "TBot");
                $res = $connection->query("SELECT * FROM Admin;");
                if ($res->num_rows == 0) {
                    $connection->query("INSERT INTO Admin (user_name, password) VALUES ('" . $_POST["username"] . "','" . md5($_POST["password"]) . "')");
                    $connection->close();
                    $_SESSION["logged"] = true;
                    header("Location: bot_manager.php");
                } else {
                    $array = $res->fetch_array(MYSQLI_ASSOC);
                    if ($_POST["username"] == $array["user_name"] && md5($_POST["password"]) == $array["password"]) {
                        $_SESSION["logged"] = true;
                        header("Location: bot_manager.php");
                    } else {
                        echo "Credenziali errate";
                    }
                }
            } catch (Exception $e) {
                echo "Error: " . $e->getMessage();
            }
        }
        ?>
    </main>



</body>

</html>