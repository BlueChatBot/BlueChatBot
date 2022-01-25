class BlueChatBot {
    constructor() {
        this.dynamic = {
            startChat: document.querySelector('.chatbox__button'),
            responseOnscreen: document.querySelector('.chatbox__support'),
            sendMsg: document.querySelector('.send__button')
        }
        this.state = false;   // state of chatbox
        this.textMessage = []; 
    }

    display() { 
        const {startChat, responseOnscreen, sendMsg} = this.dynamic;
       
        startChat.addEventListener('click', () => this.toggleState(responseOnscreen))
        sendMsg.addEventListener('click', () => this.sendMessage(responseOnscreen))
        
        const node = responseOnscreen.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.sendMessage(responseOnscreen);
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if(this.state) {
            chatbox.classList.add('chatbox--active')
        } else {
            chatbox.classList.remove('chatbox--active')
        }
    }

    sendMessage(chatbox) {
        var inputText = chatbox.querySelector('input');
        let message1 = inputText.value; 
        if (message1 === "") {
            return;
        }
        let msg1 = { name: "User", message: message1 }
        this.textMessage.push(msg1);
        // fetch from database
        // MODIFIED FOR HEROKU
        fetch('http://127.0.0.1:5000/post', {
            method: 'POST',
            // body: JSON.stringify({ message: message1 }),
            body: JSON.stringify(msg1),
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
       
          .then(data => data.json())
          .then(data => {
            let msg2 = { name: "BlueChatBot", message: data.answer };
            this.textMessage.push(msg2);
            this.updateChatText(chatbox)
            inputText.value = ''

        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(chatbox) 
            inputText.value = ''
          });
    }
    
    updateChatText(chatbox) {
        var text = '';
        this.textMessage.slice().reverse().forEach(function(item, index) {
            if (item.name === "BlueChatBot")
            {
                text += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
            }
            else
            {
                text += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
            }
          });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = text;
    }
}

const chatbox = new BlueChatBot();
chatbox.display();