import { Injectable } from '@angular/core';
import { HttpClient, HttpParams, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  public auth: string;
  public user: string;

  constructor(private http: HttpClient) {
    if (sessionStorage.getItem("auth")) {
      this.auth = sessionStorage.getItem("auth");
    }
    else {
      this.auth = '';
    }

    if (sessionStorage.getItem("user")) {
      this.user = sessionStorage.getItem("user");
    }
    else {
      this.user = '';
    }
  }

  getProducts(status:string, sort:string){
    return this.http.get('http://localhost:8765/observatory/api/products?status='+status+'&sort='+sort)
  }

  getProduct(id:string){
    return this.http.get('http://localhost:8765/observatory/api/products/'+id)
  }

  postProducts(Name:string, Description:string, Category:string, Withdrawn:string, Tags:string){
    let params = new HttpParams()
      .append('name', Name)
      .append('description', Description)
      .append('category', Category)
      .append('withdrawn', Withdrawn)
      .append('tags', Tags);
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
      .set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.post('http://localhost:8765/observatory/api/products', params.toString(), { headers: headers })
  }

  putProduct(Id:string, Name:string, Description:string, Category:string, Withdrawn:string, Tags:string){
    let params = new HttpParams()
      .append('name', Name)
      .append('description', Description)
      .append('category', Category)
      .append('withdrawn', Withdrawn)
      .append('tags', Tags);
      let headers = new HttpHeaders()
        .set("Content-Type", "application/x-www-form-urlencoded")
        .set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.put('http://localhost:8765/observatory/api/products/'+Id, params.toString(), { headers: headers })
  }

  deleteProduct(id:string){
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
      .set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.delete('http://localhost:8765/observatory/api/products/'+id, {headers: headers})
  }

  /* Log in/out down below */
  postLogin(username:string, password:string){
    let params = new HttpParams()
      .append('username', username)
      .append('password', password);
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
      .set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.post('http://localhost:8765/observatory/api/login', params.toString(), { headers: headers })
  }

  postLogout(){
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
      .set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.post('http://localhost:8765/observatory/api/logout', null, { headers: headers })
  }

  /* Shops down below */
  getShops(status:string, sort:string){
    return this.http.get('http://localhost:8765/observatory/api/shops?status='+status+'&sort='+sort)
  }

  getShop(id:string){
    return this.http.get('http://localhost:8765/observatory/api/shops/'+id)
  }

  postShops(Name:string, Address:string, Lng:string, Lat:string, Withdrawn:string, Tags:string){
    let params = new HttpParams()
      .append('name', Name)
      .append('address', Address)
      .append('lng', Lng)
	    .append('lat', Lat)
      .append('withdrawn', Withdrawn)
	    .append('tags', Tags);
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
			.set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.post('http://localhost:8765/observatory/api/shops', params.toString(), { headers: headers })
  }

  putShop(Id:string, Name:string, Address:string, Lng:string, Lat:string, Withdrawn:string, Tags:string){
    let params = new HttpParams()
      .append('name', Name)
      .append('address', Address)
      .append('lng', Lng)
	    .append('lat', Lat)
      .append('withdrawn', Withdrawn)
	    .append('tags', Tags);
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
			.set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.put('http://localhost:8765/observatory/api/shops/'+Id, params.toString(), { headers: headers })
  }

  deleteShop(id:string){
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
			.set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.delete('http://localhost:8765/observatory/api/shops/'+id, {headers: headers})
  }

  /* Prices down below */
  getPrices(tags:string, dateFrom:string, dateTo:string, distance:string, longtitude:string, latitude:string, shops:string[], products:string[], sort:string, count:string) {
    let prices_query:string;
    prices_query = 'http://localhost:8765/observatory/api/prices?shops=';
    if (shops.length != 0) {
      prices_query += shops[0];
    }
    for (var i=1; i<shops.length; i++) {
      prices_query += '&shops=' + shops[i];
    }
    for (var j=0; j<products.length; j++) {
      prices_query += '&products=' + products[j];
    }
    if (tags != '') {
      prices_query += '&tags=' + tags;
    }
    if (dateFrom != '' && dateTo != '') {
      prices_query += '&dateFrom=' + dateFrom + '&dateTo=' + dateTo;
    }
    if (longtitude != '' && latitude != '') {
      prices_query += '&geoLng=' + longtitude + '&geoLat=' + latitude;
    }
    if (distance != '') {
      prices_query += '&geoDist=' + distance;
    }
    prices_query += '&sort=' + sort;
    prices_query += '&count=100';// + count;

    return this.http.get(prices_query)
  }

  postPrices(price:string, dateFrom:string, dateTo:string, productId:string, shopId:string) {
    let params = new HttpParams()
      .append('price', price)
      .append('dateFrom', dateFrom)
      .append('dateTo', dateTo)
	    .append('productId', productId)
      .append('shopId', shopId);
    let headers = new HttpHeaders()
      .set("Content-Type", "application/x-www-form-urlencoded")
			.set("X-OBSERVATORY-AUTH", this.auth);
    return this.http.post('http://localhost:8765/observatory/api/prices', params.toString(), { headers: headers })
  }

}
