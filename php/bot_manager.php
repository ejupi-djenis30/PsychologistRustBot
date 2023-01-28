<?php
    session_start();
    if(!isset($_SESSION["logged"]) && !$_SESSION["logged"]) {
        header('Location: login.php');
        exit;
    }
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/@picocss/pico@1.*/css/pico.min.css">
    <title>Bot Manager</title>
</head>
<body>
    <main class="container">
    <h1 style="text-align:center;">Chats</h1>
    <form method="POST">
        <label for="user">User</label>
        <input type="text" name="user" placeholder="name/id" required>
        <br>
        <div class="grid">
            <label for="msgdatada">
                Data da
                <input type="datetime-local" name="msgdatada_inizio" max="<?= date("Y-m-d h:i"); ?>"required>
            </label>
            

            <label for="msgdataa">
                a
                <input type="datetime-local" name="msgdata_fine" max="<?= date("Y-m-d h:i"); ?>" required>
            </label>
        </div>
        <br>
        <input type="submit" name="cerca" value="cerca">
    </form>
    

    <?php
        if(isset($_POST["cerca"])) {
            try {
                $connection = new mysqli("localhost", "root", "", "TBot");
            $query = "SELECT prompt,response,msgtime FROM Chats WHERE ";
            $int = ctype_digit($_POST["user"]) ? intval($_POST["user"]) : null;
            if($int == null) {
                $query .= "user_name = '". $_POST["user"]. "'";
            } else {
                $query .= "user_id = ". $int;
            }

            $data_inizio = new DateTime($_POST["msgdatada_inizio"]);
            $data_fine = new DateTime($_POST["msgdata_fine"]);
            $query .= " AND msgtime >= '". $data_inizio->format("Y-m-d H:i:s"). "' AND msgtime <= '". $data_fine->format("Y-m-d H:i:s") . "';";
            if($data_inizio < $data_fine) {
                $ris = $connection->query($query);
            } else {
                throw new Exception("Le date non sono valide");
            }
            if($ris->num_rows > 0) {
                echo "<table role='grid'><tr><th>".$_POST["user"]."</th><th>Bot</th><th>Time</th></tr>";
                while(($row = $ris->fetch_assoc())) {
                    echo "<tr><td>".$row["prompt"]."</td><td>".$row["response"]."</td><td>".$row["msgtime"]."</td></tr>";
                }
                echo "</table>";
            }
            } catch (Exception $e) {
                echo "ERROR:" . $e->getMessage();
            }
            
        }
    ?>
    </main>
</body>
</html>