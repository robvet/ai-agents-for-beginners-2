<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "f0ce2d470f3efad6f8c7df376f416a4b",
  "translation_date": "2025-05-20T08:54:50+00:00",
  "source_file": "00-course-setup/AzureSearch.md",
  "language_code": "es"
}
-->
# Guía de configuración de Azure AI Search

Esta guía te ayudará a configurar Azure AI Search utilizando el portal de Azure. Sigue los pasos a continuación para crear y configurar tu servicio de Azure AI Search.

## Requisitos previos

Antes de comenzar, asegúrate de tener lo siguiente:

- Una suscripción de Azure. Si no tienes una suscripción de Azure, puedes crear una cuenta gratuita en [Azure Free Account](https://azure.microsoft.com/free/?wt.mc_id=studentamb_258691).

## Paso 1: Crear un servicio de Azure AI Search

1. Inicia sesión en el [portal de Azure](https://portal.azure.com/?wt.mc_id=studentamb_258691).
2. En el panel de navegación izquierdo, haz clic en **Create a resource**.
3. En el cuadro de búsqueda, escribe "Azure AI Search" y selecciona **Azure AI Search** de la lista de resultados.
4. Haz clic en el botón **Create**.
5. En la pestaña **Basics**, proporciona la siguiente información:
   - **Subscription**: Selecciona tu suscripción de Azure.
   - **Resource group**: Crea un nuevo grupo de recursos o selecciona uno existente.
   - **Resource name**: Ingresa un nombre único para tu servicio de búsqueda.
   - **Region**: Selecciona la región más cercana a tus usuarios.
   - **Pricing tier**: Elige un nivel de precios que se adapte a tus necesidades. Puedes comenzar con el nivel Free para pruebas.
6. Haz clic en **Review + create**.
7. Revisa la configuración y haz clic en **Create** para crear el servicio de búsqueda.

## Paso 2: Comenzar con Azure AI Search

1. Una vez que la implementación haya finalizado, navega a tu servicio de búsqueda en el portal de Azure.
2. En la página de resumen del servicio de búsqueda, haz clic en el botón **Quickstart**.
3. Sigue los pasos de la guía Quickstart para crear un índice, subir datos y realizar una consulta de búsqueda.

### Crear un índice

1. En la guía Quickstart, haz clic en **Create an index**.
2. Define el esquema del índice especificando los campos y sus atributos (por ejemplo, tipo de dato, searchable, filterable).
3. Haz clic en **Create** para crear el índice.

### Subir datos

1. En la guía Quickstart, haz clic en **Upload data**.
2. Elige una fuente de datos (por ejemplo, Azure Blob Storage, Azure SQL Database) y proporciona los detalles de conexión necesarios.
3. Asocia los campos de la fuente de datos con los campos del índice.
4. Haz clic en **Submit** para subir los datos al índice.

### Realizar una consulta de búsqueda

1. En la guía Quickstart, haz clic en **Search explorer**.
2. Ingresa una consulta de búsqueda en el cuadro para probar la funcionalidad de búsqueda.
3. Revisa los resultados y ajusta el esquema del índice o los datos según sea necesario.

## Paso 3: Usar las herramientas de Azure AI Search

Azure AI Search se integra con diversas herramientas para mejorar tus capacidades de búsqueda. Puedes usar Azure CLI, Python SDK y otras herramientas para configuraciones y operaciones avanzadas.

### Uso de Azure CLI

1. Instala Azure CLI siguiendo las instrucciones en [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?wt.mc_id=studentamb_258691).
2. Inicia sesión en Azure CLI usando el comando:
   ```bash
   az login
   ```
3. Crea un nuevo servicio de búsqueda usando Azure CLI:
   ```bash
   az search service create --resource-group <resource-group> --name <service-name> --sku Free
   ```
4. Crea un índice usando Azure CLI:
   ```bash
   az search index create --service-name <service-name> --name <index-name> --fields "field1:type, field2:type"
   ```

### Uso del Python SDK

1. Instala la biblioteca cliente de Azure Cognitive Search para Python:
   ```bash
   pip install azure-search-documents
   ```
2. Usa el siguiente código en Python para crear un índice y subir documentos:
   ```python
   from azure.core.credentials import AzureKeyCredential
   from azure.search.documents import SearchClient
   from azure.search.documents.indexes import SearchIndexClient
   from azure.search.documents.indexes.models import SearchIndex, SimpleField, edm

   service_endpoint = "https://<service-name>.search.windows.net"
   api_key = "<api-key>"

   index_client = SearchIndexClient(service_endpoint, AzureKeyCredential(api_key))

   fields = [
       SimpleField(name="id", type=edm.String, key=True),
       SimpleField(name="content", type=edm.String, searchable=True),
   ]

   index = SearchIndex(name="sample-index", fields=fields)

   index_client.create_index(index)

   search_client = SearchClient(service_endpoint, "sample-index", AzureKeyCredential(api_key))

   documents = [
       {"id": "1", "content": "Hello world"},
       {"id": "2", "content": "Azure Cognitive Search"}
   ]

   search_client.upload_documents(documents)
   ```

Para obtener información más detallada, consulta la siguiente documentación:

- [Create an Azure Cognitive Search service](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal?wt.mc_id=studentamb_258691)
- [Get started with Azure Cognitive Search](https://learn.microsoft.com/en-us/azure/search/search-get-started-portal?wt.mc_id=studentamb_258691)
- [Azure AI Search Tools](https://learn.microsoft.com/en-us/azure/ai-services/agents/how-to/tools/azure-ai-search?tabs=azurecli%2Cpython&pivots=code-examples?wt.mc_id=studentamb_258691)

## Conclusión

Has configurado correctamente Azure AI Search utilizando el portal de Azure y las herramientas integradas. Ahora puedes explorar características y capacidades más avanzadas de Azure AI Search para mejorar tus soluciones de búsqueda.

Para obtener más ayuda, visita la [documentación de Azure Cognitive Search](https://learn.microsoft.com/en-us/azure/search/?wt.mc_id=studentamb_258691).

**Aviso legal**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda la traducción profesional realizada por un humano. No nos hacemos responsables de malentendidos o interpretaciones erróneas derivadas del uso de esta traducción.