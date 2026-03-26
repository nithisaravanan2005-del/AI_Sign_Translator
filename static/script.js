let paragraph = "";
let lastWord = "";
let lastTime = 0;

// 🔊 VOICE ENGINE
let synth = window.speechSynthesis;


// 🔥 LIVE DETECTION
setInterval(() => {
    fetch("/get_prediction")
    .then(res => res.json())
    .then(data => {

        let word = data.prediction;
        document.getElementById("text").innerText = word;

        let now = Date.now();

        // ✅ ADD WORD WITH SPACE + DELAY
        if(word !== "" && word !== lastWord && (now - lastTime > 1500)){

            paragraph += word + " ";
            document.getElementById("paragraph").innerText = paragraph;

            speak(word);   // 🔊 voice

            lastWord = word;
            lastTime = now;
        }
    });
}, 300);


// 🔊 SPEAK FUNCTION (NO SPAM)
function speak(text){
    if(synth.speaking){
        synth.cancel();
    }

    let utter = new SpeechSynthesisUtterance(text);
    synth.speak(utter);
}


// ▶ START
function start(){
    fetch("/start");
}

// ⏹ STOP
function stop(){
    fetch("/stop");
}


// 🧹 CLEAR
function clearText(){
    paragraph = "";
    lastWord = "";
    document.getElementById("paragraph").innerText = "";
}


// 🚀 TRAIN
function train(){
    let name = document.getElementById("name").value;

    if(!name){
        alert("Enter sign name");
        return;
    }

    let countdownDiv = document.getElementById("countdown");
    countdownDiv.style.display = "block";

    let seconds = 5;

    countdownDiv.innerText = "Get Ready...";

    setTimeout(() => {

        fetch("/train_custom", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({sign: name})
        });

        let interval = setInterval(() => {
            countdownDiv.innerText = "Training: " + seconds + "s";

            seconds--;

            if(seconds < 0){
                clearInterval(interval);

                countdownDiv.style.display = "none";

                showPopup();
                loadCustom();
            }

        }, 1000);

    }, 1000);
}


// 🔥 POPUP
function showPopup(){
    let popup = document.getElementById("popup");
    popup.style.display = "block";

    setTimeout(()=>{
        popup.style.display = "none";
    }, 2000);
}


// 📦 LOAD CUSTOM
function loadCustom(){
    fetch("/get_custom")
    .then(res => res.json())
    .then(data => {

        let list = document.getElementById("customList");
        list.innerHTML = "";

        data.forEach(sign => {
            let li = document.createElement("li");

            li.innerHTML = `
                ${sign}
                <button onclick="deleteSign('${sign}')">❌</button>
            `;

            list.appendChild(li);
        });
    });
}

function copyText(){
    let text = document.getElementById("paragraph").innerText;

    if(text.trim() === ""){
        alert("Nothing to copy");
        return;
    }

    navigator.clipboard.writeText(text)
    .then(() => {
        alert("Copied to clipboard ✅");
    })
    .catch(() => {
        alert("Copy failed ❌");
    });
}


// ❌ DELETE
function deleteSign(sign){
    fetch("/delete_custom", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({sign: sign})
    }).then(loadCustom);
}


// INIT
loadCustom();