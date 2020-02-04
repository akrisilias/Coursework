package gr.ntua.ece.softeng18b.client.rest

import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.impl.client.HttpClients

class DefaultClientFactory implements ClientFactory {

    @Override
    CloseableHttpClient newClient() {
        return HttpClients.createDefault()
    }
}
