import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import Map from 'ol/Map';
import Tile from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import View from 'ol/View';
import { toLonLat } from 'ol/proj.js';
import { fromLonLat } from 'ol/proj.js';
import VectorSource from 'ol/source/Vector';
import {Icon, Style} from 'ol/style';
import Point from 'ol/geom/Point';
import Feature from 'ol/Feature';
import {Tile as TileLayer, Vector as VectorLayer} from 'ol/layer';

@Component({
  selector: 'app-prices',
  templateUrl: './prices.component.html',
  styleUrls: ['./prices.component.scss']
})
export class PricesComponent implements OnInit {

  prices: any;

  Longtitude: string;
  Latitude: string;
  Distance: string;
  DateFrom: string;
  DateTo: string;
  Sort: string;
  Tags: string;
  Tags2: string;
  Count: string;
  Shops: string[] = [];
  Products: string[] = [];
  ShopsLng: string[] = [];
  ShopsLat: string[] = [];
  i;
  feat;

  logged: boolean = false;
  found: boolean = false;

  view: number;
  sort: string;

  map;
  vectorSource;
  vectorLayer;

  constructor(private data: DataService) { }

  ngOnInit() {

    if (this.data.auth != "") {
      this.logged = true;
    }
    else {
      this.logged = false;
    }

    this.sort = 'price|ASC';

    this.Count = '20';
    this.Tags = '';
    this.Sort = 'price|ASC';
    this.DateTo = '';
    this.DateFrom = '';
    this.Distance = '';
    this.Latitude = '';
    this.Longtitude = '';

    if (this.data.auth != "") {
      this.logged = true;
    }
    else {
      this.logged = false;
    }

    this.data.getShops('ACTIVE', 'id|DESC')
      .subscribe(data => {
        for (var i=0; i<(data as any).shops.length; i++) {
          this.Shops.push((data as any).shops[i].id)
          console.log((data as any).shops[i].id);
		  this.ShopsLng.push((data as any).shops[i].lng)
          console.log(this.ShopsLng[i]);
		  this.ShopsLat.push((data as any).shops[i].lat)
          console.log(this.ShopsLat[i]);
        }
        console.log(this.Shops);
		this.initilizeMap();
      }
    );

    this.data.getProducts('ACTIVE', 'id|DESC')
      .subscribe(data => {
        for (var i=0; i<(data as any).products.length; i++) {
          this.Products.push((data as any).products[i].id)
          console.log((data as any).products[i].id);
        }
        console.log(this.Products);

      }
    );

    this.view = 20;

  }

  getClick() {
    console.log("Tags");
    console.log(this.Tags.toLowerCase().replace(/\s/g,","));
    console.log("DateFrom");
    console.log(this.DateFrom);
    console.log("DateTo");
    console.log(this.DateTo);
    console.log("Distance");
    console.log(this.Distance);
    console.log("Long");
    console.log(this.Longtitude);
    console.log("Lat");
    console.log(this.Latitude);
    console.log("Shops");
    console.log(this.Shops);
    console.log("Products");
    console.log(this.Products);

    this.Tags2 = this.Tags;
    this.Tags2 = this.Tags2.toLowerCase().replace(/\s/g,",");

    this.data.getPrices(this.Tags2, this.DateFrom, this.DateTo, this.Distance, this.Longtitude, this.Latitude, this.Shops, this.Products, this.sort, this.Count)
      .subscribe(data => {
        this.prices = data;
        console.log(this.prices);
        }
      );

    this.found = true;

  }

  sortClick() {
    console.log(this.sort);
    this.data.getPrices(this.Tags2, this.DateFrom, this.DateTo, this.Distance, this.Longtitude, this.Latitude, this.Shops, this.Products, this.sort, this.Count)
      .subscribe(data => {
        this.prices = data;
        console.log(this.prices);
        }
      );
  }

  initilizeMap() {

	var coord = fromLonLat([23.71622, 37.97945]);
	this.i=1;
	this.vectorSource = new VectorSource();
	console.log(this.ShopsLat.length);

	for (var i=0; i<this.ShopsLat.length; i++) {

		var [coord1, coord2]=fromLonLat([this.ShopsLng[i], this.ShopsLat[i]]);
		let feature = new Feature(
            new Point([coord1, coord2])
          );
		this.feat=feature;

          let style= new Style({
			image: new Icon(({
			color: 'yellow',
			crossOrigin: 'anonymous',
			src: 'assets/img/dot.png',
			imgSize: [20, 20]
			}))
		})

          // let value = parseFloat(datum.value);
          feature.setStyle(style);
          this.vectorSource.addFeature(feature);

		  console.log("AEK");

        }


    this.vectorLayer = new VectorLayer({
      source: this.vectorSource
    });

    this.map = new Map({
      target: 'map',
      layers: [
        new TileLayer({ source: new OSM() }),
        this.vectorLayer
      ],
      view: new View({
        center: coord,
        zoom: 10
      })
    });

	this.map.on('click', (evt) => {
		if (this.i==1)
			this.i=0;
		else
		   this.vectorSource.removeFeature(this.feat);


		//this.vectorSource.clear();
		var [coord1, coord2]=this.map.getEventCoordinate(evt.originalEvent);

		var lonlat = toLonLat([coord1, coord2]);
		var lon = lonlat[0];
		var lat = lonlat[1];
		this.Longtitude=lon;
		this.Latitude=lat;
		// …
			console.log('clicked');


          this.feat = new Feature(
            new Point([coord1, coord2])
          );

          let style= new Style({
			image: new Icon(({
			color: 'red',
			crossOrigin: 'anonymous',
			src: 'assets/img/dot.png',
			imgSize: [20, 20]
			}))
		})

          // let value = parseFloat(datum.value);
          this.feat.setStyle(style);
          this.vectorSource.addFeature(this.feat);

	});

  }


}
