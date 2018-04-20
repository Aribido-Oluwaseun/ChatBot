var messages = [], //array that hold the record of each string in chat
  lastUserMessage = "", //keeps track of the most recent input string from the user
  botMessage = "", //var keeps track of what the chatbot is going to say
  botName = 'TieBot' //name of the chatbot

//function to provide answer/response from the bot
function chatbotResponse() {
  
  botMessage = "I'm confused"; //the default message
 
  if (lastUserMessage === 'hi' || lastUserMessage =='hello') {
    const hi = ['hi','howdy','hello']
    botMessage = hi[Math.floor(Math.random()*(hi.length))];;     
  }
  if (lastUserMessage === 'name') {
    botMessage = 'My name is ' + botName;
  }
  else{
	getAnswer();	
  }
}

function getAnswer(){
$.ajax({
async: false,
type: "get",
url: '/question',
success: function(response){
returnAns(response);
}
});
}
//callback function to get the response
function returnAns(response){
botMessage = response;
}

/*
//JSON functions
function make_JSON_user_message(lastUserMessage) {//serialize data function

  var returnArray = {};
 
    returnArray['user_message'] = lastUserMessage;

  return returnArray;
}

function make_JSON_answer(userAnswer) {//serialize data function

  var returnArray = {};
 
    returnArray['user_message'] = lastUserMessage;

  return returnArray;
}
*/
//new question asked
function newEntry() {
  //if the message from the user isn't empty then run
  if (document.getElementById("chatbox").value != "") {
    //pulls the value from the chatbox ands sets it to lastUserMessage
    lastUserMessage = document.getElementById("chatbox").value;
	$.post( "/question",{
	question:document.getElementById("chatbox").value
	});

    //sets the chat box to be clear
    document.getElementById("chatbox").value = "";
    document.getElementById("chatlog").innerHTML += "<p id='you'> You  : " + lastUserMessage + "</p>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";
    //sets the variable botMessage in response to lastUserMessage
    chatbotResponse();
    document.getElementById("chatlog").innerHTML +="<p id='bot'>" + botName + ": " + botMessage + "</p>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";
    document.getElementById("chatlog").innerHTML += "<br>";

  
  }
}

//send the feedback to backend database
function send_feedback(){
$.post( "/feedback",{
feedback:document.getElementById("textbox").value,
success: function(response){
}
});
}

//runs the keypress() function when a key is pressed
document.onkeypress = keyPress;
//if the key pressed is 'enter' runs the function newEntry()
function keyPress(e) {
  var x = e || window.event;
  var key = (x.keyCode || x.which);
  if (key == 13 || key == 3) {
  
    //runs this function when enter is pressed
    newEntry();
  }
}

//clears the placeholder text ion the chatbox
//this function is set to run when the users brings focus to the chatbox, by clicking on it
function placeHolder() {
  document.getElementById("chatbox").placeholder = "";
}


