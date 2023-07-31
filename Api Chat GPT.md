Nestjs: angular de backend
Yanm mas compatibles con servidores
npm version mas rapida

```
npm i -g @nestjs/cli // instalar el clic

nest new example // crear el proyecto
```

<h5>Controladores</h5>
Escucha solicitudes y envia respuestas
```
nest g controller cats
```

Con postman se puede acceder

```
Para acceder en postman
http://localhost:3000/cats
```

```JS
import { Controller } from '@nestjs/common';

@Controller('cats')

export class CatsController {}
```

Para que apareza en el postaman cats cambiamos:

```JS
import { Controller , Get} from '@nestjs/common';

  

@Controller('cats')

export class CatsController {

    @Get()

    findAll():string{

        return 'Cats Controller Get Succesfull';

    }

}
```
EndPoint: Salidas del Controller
Codigos status:
Informational
Succesful
Redirection
Client Error
Server Error

Para hacer un post:
```JS
import { Controller , Get , Post} from '@nestjs/common';

  

@Controller('cats')

export class CatsController {

    @Post()

    create():string{

        return 'Agrega un POST';

    }

    @Get()

    findAll():string{

        return 'Cats Controller Get Succesfull';

    }

}
```

Operaciones CRUD
Create
Read
Unite
Delete

```c
npx nest generate resource
```

Postman:
```
http://localhost:3000/creatures/646
```

Para conectar a chatGPT:

```python
/**

 * This code demonstrates how to use the OpenAI API to generate chat completions.

 * The generated completions are received as a stream of data from the API and the

 * code includes functionality to handle errors and abort requests using an AbortController.

 * The API_KEY variable needs to be updated with the appropriate value from OpenAI for successful API communication.

 */

  

const API_URL = "https://api.openai.com/v1/chat/completions";

const API_KEY = "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";

  

const promptInput = document.getElementById("promptInput");

const generateBtn = document.getElementById("generateBtn");

const stopBtn = document.getElementById("stopBtn");

const resultText = document.getElementById("resultText");

  

let controller = null; // Store the AbortController instance

  

const generate = async () => {

  // Alert the user if no prompt value

  if (!promptInput.value) {

    alert("Please enter a prompt.");

    return;

  }

  
  

  // Disable the generate button and enable the stop button

  generateBtn.disabled = true;

  stopBtn.disabled = false;

  resultText.innerText = "Generating...";

  

  // Create a new AbortController instance

  controller = new AbortController();

  const signal = controller.signal;

  

  try {

    // Fetch the response from the OpenAI API with the signal from AbortController

    const response = await fetch(API_URL, {

      method: "POST",

      headers: {

        "Content-Type": "application/json",

        Authorization: `Bearer ${API_KEY}`,

      },

      body: JSON.stringify({

        model: "gpt-3.5-turbo",

          messages: [{ role: "user", content:'Responds ironically and like a know-it-all and always adds something about why Javascript is the best language in the world. '+ promptInput.value

        }],

        max_tokens: 100,

        stream: true, // For streaming responses

      }),

      signal, // Pass the signal to the fetch request

    });

    console.log({

      method: "POST",

      headers: {

        "Content-Type": "application/json",

        Authorization: `Bearer ${API_KEY}`,

      },

      body: JSON.stringify({

        model: "gpt-3.5-turbo",

        messages: [{ role: "user", content:'Responds ironically and like a know-it-all and always adds something about why Javascript is the best language in the world. '+ promptInput.value

      }],

        max_tokens: 100,

        stream: true, // For streaming responses

      }),

      signal, // Pass the signal to the fetch request

    });

  
  
  

    // Read the response as a stream of data

    const reader = response.body.getReader();

    const decoder = new TextDecoder("utf-8");

    resultText.innerText = "";

  

    while (true) {

      const { done, value } = await reader.read();

      if (done) {

        break;

      }

      // Massage and parse the chunk of data

      const chunk = decoder.decode(value);

      const lines = chunk.split("\n");

      const parsedLines = lines

        .map((line) => line.replace(/^data: /, "").trim()) // Remove the "data: " prefix

        .filter((line) => line !== "" && line !== "[DONE]") // Remove empty lines and "[DONE]"

        .map((line) => JSON.parse(line)); // Parse the JSON string

  

      for (const parsedLine of parsedLines) {

        const { choices } = parsedLine;

        const { delta } = choices[0];

        const { content } = delta;

        // Update the UI with the new content

        if (content) {

          resultText.innerText += content;

        }

      }

    }

  } catch (error) {

    // Handle fetch request errors

    if (signal.aborted) {

      resultText.innerText = "Request aborted.";

    } else {

      console.error("Error:", error);

      resultText.innerText = "Error occurred while generating.";

    }

  } finally {

    // Enable the generate button and disable the stop button

    generateBtn.disabled = false;

    stopBtn.disabled = true;

    controller = null; // Reset the AbortController instance

  }

};

  

const stop = () => {

  // Abort the fetch request by calling abort() on the AbortController instance

  if (controller) {

    controller.abort();

    controller = null;

  }

};

  

promptInput.addEventListener("keyup", (event) => {

  if (event.key === "Enter") {

    generate();

  }

});

generateBtn.addEventListener("click", generate);

stopBtn.addEventListener("click", stop);
```

![[Pasted image 20230727110158.png]]
![[Pasted image 20230727110228.png]]
![[Pasted image 20230727110252.png]]

