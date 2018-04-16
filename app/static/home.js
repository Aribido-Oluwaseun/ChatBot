nlp = window.nlp_compromise;
var messages = [], //array that hold the record of each string in chat
  lastUserMessage = "", //keeps track of the most recent input string from the user
  botMessage = "", //var keeps track of what the chatbot is going to say
  botName = 'TieBot', //name of the chatbot
  talking = true; //when false the speach function doesn't work
  //response by bot

function chatbotResponse() {
  talking = true;
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
//botMessage= document.getElementById("test").innerHTML
              
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
function returnAns(response){
botMessage = response;
}
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


function write_to_file()
{
var fso = new ActiveXObject("Scripting.FileSystemObject");
var s = fso.CreateTextFile("data.txt", true);
s.WriteLine(lastUserMessage);
s.Close();

}
function newEntry() {
  //if the message from the user isn't empty then run
  if (document.getElementById("chatbox").value != "") {
    //pulls the value from the chatbox ands sets it to lastUserMessage
    lastUserMessage = document.getElementById("chatbox").value;
	$.post( "/question",{
question:document.getElementById("chatbox").value,
success: function(answer){

}
});

 //write to a txt file
    //sets the chat box to be clear
    document.getElementById("chatbox").value = "";
    //adds the value of the chatbox to the array messages
    messages.push("<p id='you'> You  : " + lastUserMessage + "</p><br>");
    //Speech(lastUserMessage);  //says what the user typed outloud
    //sets the variable botMessage in response to lastUserMessage
    chatbotResponse();
    //add the chatbot's name and message to the array messages
	
	
    messages.push("<p id='bot'>" + botName + ": " + botMessage + "</p>");
    // says the message using the text to speech function written below
    //Speech(botMessage);
    //outputs the last few array elements of messages to html
    for (var i = 1; i < 8; i++) {
      if (messages[messages.length - i])
        document.getElementById("chatlog" + i).innerHTML = messages[messages.length - i];
    }
  }
}


function print_returnArray(returnArray){
console.log(returnArray);
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
  if (key == 38) {
    console.log('hi')
      //document.getElementById("chatbox").value = lastUserMessage;
  }
}
//clears the placeholder text ion the chatbox
//this function is set to run when the users brings focus to the chatbox, by clicking on it
function placeHolder() {
  document.getElementById("chatbox").placeholder = "";
}


