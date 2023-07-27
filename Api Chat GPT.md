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