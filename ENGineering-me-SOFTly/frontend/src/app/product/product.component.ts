import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import { ActivatedRoute, Router } from '@angular/router';
import { ProductsComponent } from '../products/products.component';

@Component({
  selector: 'app-product',
  templateUrl: './product.component.html',
  styleUrls: ['./product.component.scss']
})
export class ProductComponent implements OnInit {

  product: any;

  id: string;

  Name: string;
  Description: string;
  Category: string;
  Withdrawn: string;
  Tags: string;

  constructor(private data: DataService, private route: ActivatedRoute, private router: Router) { }

  ngOnInit() {
    this.id = this.route.snapshot.paramMap.get('id');

    this.data.getProduct(this.id)
      .subscribe(data => {
        this.product = data;
        console.log(this.product);
        this.Name = this.product.name;
        this.Description = this.product.description;
        this.Category = this.product.category;
        this.Withdrawn = this.product.withdrawn;
        this.Tags = this.product.tags;
        }
      );
  }

  putClick() {
    this.data.putProduct(this.id, this.Name, this.Description, this.Category, this.Withdrawn, this.Tags)
      .subscribe( data => {
        console.log(data);
        this.router.navigate(['products']);
        }
      );

  }

}
