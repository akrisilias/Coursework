package gr.ntua.ece.softeng18b.client.rest

import org.apache.http.conn.ssl.NoopHostnameVerifier
import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.impl.client.HttpClients

import javax.net.ssl.SSLContext
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager
import java.security.SecureRandom
import java.security.cert.X509Certificate

class SSLErrorTolerantClientFactory implements ClientFactory {

    private static final SSLContext SSL = SSLContext.getInstance("SSL")

    static {
        TrustManager[] tms = new TrustManager[1]
        tms[0] = new InsecureTrustManager()
        SSL.init(null, tms, new SecureRandom())
    }

    private static class InsecureTrustManager implements X509TrustManager {

        @Override
        X509Certificate[] getAcceptedIssuers() {
            return null;
        }

        @Override
        void checkClientTrusted(X509Certificate[] certs, String authType) {  }

        @Override
        void checkServerTrusted(X509Certificate[] certs, String authType) {  }

    }

    @Override
    CloseableHttpClient newClient() {
        HttpClients.custom().setSSLHostnameVerifier(NoopHostnameVerifier.INSTANCE).setSSLContext(SSL).build()
    }

}
