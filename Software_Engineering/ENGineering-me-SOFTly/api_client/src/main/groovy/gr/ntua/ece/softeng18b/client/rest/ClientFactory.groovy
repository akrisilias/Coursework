package gr.ntua.ece.softeng18b.client.rest

import org.apache.http.impl.client.CloseableHttpClient

interface ClientFactory {
    CloseableHttpClient newClient()
}
